import sqlalchemy as sa
from sqlalchemy import orm

from models.base import BaseModel


class Message(BaseModel):
    __tablename__ = "messages"

    user_id: orm.Mapped[int] = orm.mapped_column(sa.BigInteger, sa.ForeignKey("users.id"), nullable=False)
    role: orm.Mapped[str] = orm.mapped_column(sa.String, nullable=False)
    text: orm.Mapped[str] = orm.mapped_column(sa.String, nullable=False)
    response_type: orm.Mapped[str] = orm.mapped_column(sa.String, nullable=True)