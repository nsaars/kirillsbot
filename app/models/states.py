import sqlalchemy as sa
from sqlalchemy import orm

from models.base import BaseModel


class State(BaseModel):
    __tablename__ = "states"

    user_id: orm.Mapped[int] = orm.mapped_column(sa.Integer, sa.ForeignKey("users.id"))
    title: orm.Mapped[str] = orm.mapped_column(sa.String, nullable=True)
    data: orm.Mapped[str] = orm.mapped_column(sa.JSON, nullable=True)
