from sqlalchemy import create_engine

from sqlalchemy import orm

from models.base import Base


def create_db_connection():
    DATABASE_URL = "postgresql+psycopg2://postgres:pulat@localhost:5432/postgres"
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    return orm.sessionmaker(autocommit=False, autoflush=False, bind=engine)


SessionLocal = create_db_connection()
