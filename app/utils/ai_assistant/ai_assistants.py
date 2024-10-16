import json
import os
from datetime import datetime
from pprint import pprint
from typing import List, Tuple, Dict
import pandas as pd
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from utils.ai_assistant.tools import get_tools

current_dir = os.path.dirname(os.path.abspath(__file__))


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
        pprint(similar_docs)
        return "\n\n".join(f"Вопрос:{doc.page_content}\nОтвет:{doc.metadata['answer']}" for doc in similar_docs)

    @staticmethod
    def get_formatted_datetime():
        now = datetime.now()
        weekday = now.weekday()
        days = {
            0: "понедельник", 1: "вторник", 2: "среда", 3: "четверг",
            4: "пятница", 5: "суббота", 6: "воскресенье"
        }
        formatted_date = now.strftime("%Y-%m-%d")
        formatted_time = now.strftime("%H:%M")
        if now.hour > 18:
            formatted_time = "уже" + formatted_time
        return days[weekday] + ", " + formatted_date, formatted_time

    async def get_default_response(self, text: str, history: List[Tuple[str, str]] = None, new_request = None) -> Dict:
        if history is None:
            history = []

        default_template = ChatPromptTemplate(
            [self._system_prompt]
            + history + [('user', "{message}")])

        return {'default_response': await (
                {"message": RunnablePassthrough(), 'language': RunnableLambda(lambda x : new_request['previous_language'])}
                | default_template
                | self._llm.bind(tools=get_tools(*self.get_formatted_datetime()))
        ).ainvoke(text)}

    async def get_question_response(self, text: str, history: List[Tuple[str, str]] = None, new_request = None) -> Dict:
        if history is None:
            history = []

        question_template = ChatPromptTemplate(
            [self._system_prompt] +
            history + [('user', self._prompt_templates['qa_question'] +
                        """База знаний для ответа на вопрос:\n{context}\n\nВопрос клиента: {message}""")])
        context_docs = await self._retriever.ainvoke(new_request['new_request'])

        formatted_context = RunnableLambda(lambda x: self.format_docs(context_docs))
        response = await (
                {"context": formatted_context, "message": RunnablePassthrough(), "language":RunnableLambda(lambda x : new_request['previous_language'])}
                | question_template
                | self._llm.bind(tools=get_tools(*self.get_formatted_datetime()))
        ).ainvoke(new_request['new_request'])

        return {'question_response': response}

    async def get_bad_words_response(self, text: str, history: List[Tuple[str, str]] = None, new_request = None) -> Dict:
        if history is None:
            history = []

        bad_words_template = ChatPromptTemplate(
            [self._system_prompt] +
            history + [("user", self._prompt_templates['qa_bad_words'])])

        return {'bad_words_response': await (
                {"message": RunnablePassthrough(), 'language': RunnableLambda(lambda x : new_request['previous_language'])}
                | bad_words_template
                | self._llm.bind(tools=get_tools(*self.get_formatted_datetime()))
        ).ainvoke(text)}

    async def get_humor_response(self, text: str, history: List[Tuple[str, str]] = None, new_request = None) -> Dict:
        if history is None:
            history = []

        humor_template = ChatPromptTemplate(
            [self._system_prompt] +
            history + [("user", self._prompt_templates['qa_humor'])])

        return {'humor_response': await (
                {"message": RunnablePassthrough(), 'language': RunnableLambda(lambda x : new_request['previous_language'])}
                | humor_template
                | self._llm.bind(tools=get_tools(*self.get_formatted_datetime()))
        ).ainvoke(text)}


class AiHelpers:
    def __init__(self, llm: str = "gpt-4o-mini",
                 prompt_templates_file_path: str = os.path.join(current_dir, "helpers_prompt_templates.json")):
        self._llm = ChatOpenAI(model=llm)

        with open(prompt_templates_file_path, "r", encoding='utf-8') as f:
            self._prompt_templates = json.load(f)

    async def get_message_type(self, text: str, history: List[Tuple[str, str]] = None) -> Dict:
        prompt_template = ChatPromptTemplate(
            history + [("system", self._prompt_templates['type_detector']), ("user", "{message}")]
        )
        return {'type': await ({"message": RunnablePassthrough()}
                               | prompt_template
                               | self._llm
                               ).ainvoke(text)}

    async def get_chat_summary(self, history: List[Tuple[str, str]] = None) -> Dict:
        prompt_template = ChatPromptTemplate([("system", self._prompt_templates['summary_system']),
                                              ("user", "{message}")])
        history_text = ''.join([f'{role}\n{message}\n\n' for role, message in history])
        return {'summary': await ({"message": RunnablePassthrough()}
                                  | prompt_template
                                  | self._llm
                                  ).ainvoke(
            f"'{self._prompt_templates['summary']}\nИстория чата с клиентом:\n{history_text}'")}

    async def get_new_request(self, text: str, history: List[Tuple[str, str]]) -> Dict[str, str]:
        """
        Modify the user's input if necessary.
        """
        if not text:
            raise ValueError("Input text cannot be empty")
        if not isinstance(history, list):
            raise TypeError("History must be a list of tuples")

        prompt_template = ChatPromptTemplate.from_messages([
            MessagesPlaceholder('chat_history'),
            ('system', self._prompt_templates['default_system']),
            ('user', self._prompt_templates['change_question'])
        ])

        class RequestModel(BaseModel):
            new_request: str = Field(description="Изменённый запрос")
            previous_language: str = Field(description="Язык первоначального запроса (до изменения).")

        response = await (prompt_template
                          | self._llm.with_structured_output(RequestModel)
                          ).ainvoke({'input': text, 'chat_history': history})

        return {'new_request': response.new_request, 'previous_language': response.previous_language}

    @staticmethod
    def _format_history(history: List[Tuple[str, str]]) -> str:
        if not history:
            return ''
        return '\n'.join(f"{'Клиент' if role == 'user' else 'Ты'}:\n{message}" for role, message in history)
