import json
import os
import asyncio
from typing import List, Tuple, Dict
import pandas as pd

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

current_dir = os.path.dirname(os.path.abspath(__file__))

os.path.join(current_dir, "text-embedding-3-small")


class AiQuestionAnswering:
    def __init__(self, llm: str = "gpt-4o-mini",
                 embedding_model: str = "text-embedding-3-large",
                 data_file_path: str = os.path.join(current_dir, "data_quasar79.csv"),
                 prompt_templates_file_path: str = os.path.join(current_dir, "qa_prompt_templates.json"),
                 search_quantity: int = 3):
        self._store = LocalFileStore(os.path.join(current_dir, "cache"))
        self._underlying_embeddings = OpenAIEmbeddings(model=embedding_model)
        self._cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            self._underlying_embeddings, self._store, namespace=self._underlying_embeddings.model
        )
        self._llm = ChatOpenAI(model=llm)
        self._df = pd.read_csv(data_file_path)
        self._docs = [Document(page_content=row['question'], metadata={'answer': row['answer'], 'id': _}) for _, row in
                      self._df.iterrows()]
        self._vectorstore = Chroma.from_documents(documents=self._docs, embedding=self._cached_embedder)
        self._retriever = self._vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": search_quantity})

        with open(prompt_templates_file_path, "r", encoding='utf-8') as f:
            self._prompt_templates = json.load(f)

        self._system_prompt = ("system", self._prompt_templates['qa_system'])

    def test(self):
        print(self._vectorstore.similarity_search_with_relevance_scores(query="почему ии нужен мне?", k=3))

    @staticmethod
    def format_docs(similar_docs: List[Document]) -> str:
        return "\n\n".join(f"Вопрос:{doc.page_content}\nОтвет:{doc.metadata['answer']}" for doc in similar_docs)

    async def get_default_response(self, text: str, history: List[Tuple[str, str]] = None) -> Dict[str, str]:
        if history is None:
            history = []

        default_template = ChatPromptTemplate(
            [self._system_prompt] + history + [('user', "{message}")])

        return {'default_response': await (
                {"message": RunnablePassthrough()}
                | default_template
                | self._llm
                | StrOutputParser()
        ).ainvoke(text)}

    async def get_question_response(self, text: str, history: List[Tuple[str, str]] = None) -> Dict[str, str]:
        if history is None:
            history = []

        question_template = ChatPromptTemplate(
            [self._system_prompt] + history +
            [('user', """База знаний для ответа на вопрос:\n{context}\n\nВопрос клиента: {message}""")])

        return {'question_response': await (
                {"context": self._retriever | self.format_docs, "message": RunnablePassthrough()}
                | question_template
                | self._llm
                | StrOutputParser()
        ).ainvoke(text)}

    async def get_bad_words_response(self, text: str, history: List[Tuple[str, str]] = None) -> Dict[str, str]:
        if history is None:
            history = []

        bad_words_template = ChatPromptTemplate(
            [self._system_prompt] + history + [("user", self._prompt_templates['qa_bad_words'])])

        return {'bad_words_response': await (
                {"message": RunnablePassthrough()}
                | bad_words_template
                | self._llm
                | StrOutputParser()
        ).ainvoke(text)}

    async def get_humor_response(self, text: str, history: List[Tuple[str, str]] = None) -> Dict[str, str]:
        if history is None:
            history = []

        humor_template = ChatPromptTemplate(
            [self._system_prompt] + history + [("user", self._prompt_templates['qa_humor'])])

        return {'humor_response': await (
                {"message": RunnablePassthrough()}
                | humor_template
                | self._llm
                | StrOutputParser()
        ).ainvoke(text)}


class AiTypeDetector:
    def __init__(self, llm: str = "gpt-4o-mini",
                 prompt_templates_file_path: str = os.path.join(current_dir, "td_prompt_templates.json")):
        self._llm = ChatOpenAI(model=llm)
        with open(prompt_templates_file_path, "r", encoding='utf-8') as f:
            self._prompt_templates = json.load(f)
        self._system_prompt = ("system", self._prompt_templates['td_system'])

    async def get_response(self, text: str, history: List[Tuple[str, str]] = None) -> Dict[str, str]:
        history_string = self._format_history(history)
        prompt_template = ChatPromptTemplate(
            [self._system_prompt,
             f"История твоего чата с клиентом:\n\n{history_string}\n\nНовое сообщение клиента:\n{text}"]
        )
        return {'type': await ({"message": RunnablePassthrough()}
                               | prompt_template
                               | self._llm
                               | StrOutputParser()
                               ).ainvoke(text)}

    @staticmethod
    def _format_history(history: List[Tuple[str, str]]) -> str:
        if not history:
            return ''
        return '\n'.join(f"{'Клиент' if role == 'user' else 'Ты'}:\n{message}" for role, message in history)


class AiChain:
    decision = None
    responses = {
        'contact_response': "Я могу предоставить Вам контакт нашего Co-founder, CFO [Кирилла Добрели]"
                            "(https://t.me/BitMarkt).",
        'next_question_response': "Я бы не прочь поговорить на отвлечённые темы, но я на работе. Если у вас"
                                  " есть вопросы по поводу чат-ботов или их функциональности, буду рад помочь!"
    }
    qa = AiQuestionAnswering()
    td = AiTypeDetector()

    @classmethod
    async def get_responses(cls, text: str, history: List[Tuple[str, str]] = None):
        tasks = [
            asyncio.create_task(cls.td.get_response(text, history)),
            asyncio.create_task(cls.qa.get_question_response(text, history)),
            asyncio.create_task(cls.qa.get_default_response(text, history)),
            asyncio.create_task(cls.qa.get_bad_words_response(text, history)),
            asyncio.create_task(cls.qa.get_humor_response(text, history)),
        ]

        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            yield result

    @classmethod
    async def get_proper_response(cls, text: str, history: List[Tuple[str, str]] = None):
        cls.responses = {
            'contact_response': "Я могу предоставить Вам контакт нашего Co-founder, CFO [Кирилла Добрели]"
                                "(https://t.me/BitMarkt).",
            'next_question_response': "Я бы не прочь поговорить на отвлечённые темы, но я на работе. Если у вас"
                                      " есть вопросы по поводу чат-ботов или их функциональности, буду рад помочь!"
        }
        cls.decision = None

        async for response in cls.get_responses(text, history):
            key, message = list(response.items())[0]
            print(response)
            if key == 'type':
                if message in ('1', '4', '6'):
                    cls.decision = 'question_response'
                elif message == '3':
                    cls.decision = 'humor_response'
                elif message == '5':
                    cls.decision = 'bad_words_response'
                elif message == '7':
                    cls.decision = 'contact_response'
                    return cls.responses[cls.decision]
                elif message == '8':
                    cls.decision = 'next_question_response'
                    return cls.responses[cls.decision]
                else:
                    cls.decision = 'default_response'
            else:
                cls.responses.update(response)
                if cls.decision and cls.responses.get(cls.decision):
                    return cls.responses[cls.decision]


if __name__ == '__main__':
    qa = AiQuestionAnswering()
    qa.test()
