
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.orm import relationship

from models.base import BaseModel


class User(BaseModel):
    __tablename__ = "users"

    telegram_id: orm.Mapped[int] = orm.mapped_column(sa.BigInteger, nullable=False)
    telegram_username: orm.Mapped[str] = orm.mapped_column(sa.String, nullable=False)
    telegram_name: orm.Mapped[str] = orm.mapped_column(sa.String, nullable=False)

