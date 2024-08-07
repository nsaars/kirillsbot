import os
import pickle

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from data.config import CREDENTIALS

# Если изменяются области доступа, удалите файл token.json
SCOPES = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/documents']


def authenticate():
    """Функция для аутентификации и авторизации пользователя"""
    creds = None
    # Файл token.json хранит токен доступа пользователя
    if os.path.exists('token.json'):
        print('123123')
        with open('token.json', 'rb') as token:
            creds = pickle.load(token)
    # Если нет действительного токена, проходим процесс авторизации
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_config(CREDENTIALS, SCOPES)
            creds = flow.run_local_server(port=0)
        # Сохраняем токен для следующего использования
        with open('token.json', 'wb') as token:
            pickle.dump(creds, token)
    return creds


def create_document():
    """Функция для создания нового документа в Google Docs"""
    creds = authenticate()
    service = build('docs', 'v1', credentials=creds)

    title = 'My Document'
    body = {
        'title': title
    }
    doc = service.documents().create(body=body).execute()
    print(f'Created document with title: {doc.get("title")} and ID: {doc.get("documentId")}')


def add_text_to_document(document_id, text, index=None, service=None):
    """Функция для добавления текста на новую пустую строку в документ Google Docs"""
    if not service:
        creds = authenticate()
        service = build('docs', 'v1', credentials=creds)

    if not index:
        document = service.documents().get(documentId=document_id).execute()
        content = document.get('body').get('content', [])
        index = content[-1]['endIndex'] - 1 if content else 1

    requests = [
        {
            'insertText': {
                'location': {
                    'index': index,
                },
                'text': text + '\n'
            }
        }
    ]
    result = service.documents().batchUpdate(documentId=document_id, body={'requests': requests}).execute()
    return result
