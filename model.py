from sqlalchemy import Column, Integer, String
from database import Base


class Candles(Base):
    __tablename__ = "candles_monthly_recent3"
    timestamp = Column(Integer, primary_key=True)
    open = Column(Integer)
    high = Column(Integer)
    low = Column(Integer)
    volume = Column(Integer)
    symbol = Column(String, primary_key=True)
    duration = Column(Integer, primary_key=True)
    insertion_timestamp = Column(Integer, primary_key=True)
