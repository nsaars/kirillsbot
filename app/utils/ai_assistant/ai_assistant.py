import os
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


class AiAssistant:
    def __init__(self, llm="gpt-4o", embedding_model="text-embedding-3-small",
                 data_file_path="utils/ai_assistant/data_quasar79.csv",
                 prompt_template_file_path="utils/ai_assistant/prompt_template.txt",
                 search_quantity=3):
        self._store = LocalFileStore("utils/ai_assistant/cache")

        self._underlying_embeddings = OpenAIEmbeddings(model=embedding_model)
        self._cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            self._underlying_embeddings, self._store, namespace=self._underlying_embeddings.model
        )

        self._llm = ChatOpenAI(model=llm)

        self._df = pd.read_csv(data_file_path)
        self._docs = [Document(page_content=row['question'], metadata={'answer': row['answer']}) for _, row in
                      self._df.iterrows()]

        self._vectorstore = Chroma.from_documents(documents=self._docs, embedding=self._cached_embedder)
        self._retriever = self._vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": search_quantity})

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", open(prompt_template_file_path, "r", encoding='utf-8').read()),
            ("user", """Контекстные данные:\n{context}\n\nВопрос клиента: {question}""")
        ])

    @staticmethod
    def format_docs(similar_docs):
        return "\n\n".join(f"Вопрос:{doc.page_content}\nОтвет:{doc.metadata['answer']}" for doc in similar_docs)

    def get_response(self, text):
        return (
                {"context": self._retriever | self.format_docs, "question": RunnablePassthrough()}
                | self.prompt_template
                | self._llm
                | StrOutputParser()
        ).invoke(text)


ai = AiAssistant()
