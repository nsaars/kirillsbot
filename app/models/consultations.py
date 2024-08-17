import sqlalchemy as sa
from sqlalchemy import orm
from models.base import BaseModel


class Consultation(BaseModel):
    __tablename__ = "consultations"

    user_id: orm.Mapped[int] = orm.mapped_column(sa.BigInteger, sa.ForeignKey("users.id"), nullable=False)
    summary: orm.Mapped[str] = orm.mapped_column(sa.String, nullable=False)
    date: orm.Mapped[str] = orm.mapped_column(sa.DateTime, nullable=False)
