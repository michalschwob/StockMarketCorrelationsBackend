from fastapi import FastAPI, Depends, Request
from databases import Database
import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import aiosqlite
import json
from fastapi.middleware.cors import CORSMiddleware
import model
from database import engine, SessionLocal
from sqlalchemy.orm import Session
from aiosqlite import Connection
import timeit

app = FastAPI()

model.Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess(data_frame):
    # consolidates the date to first day of the month for easier plotting
    data_frame['timestamp_month'] = data_frame["timestamp"].astype("datetime64[M]")
    # converts the timestamp to a datetime object with showing only year and month
    data_frame['timestamp_month'] = data_frame['timestamp_month'].dt.strftime('%Y-%m')
    final_df = data_frame.pivot_table(index=["timestamp_month"], columns="symbol", values="high")
    correlation = final_df.corr()
    # Sort the columns and rows of the correlation matrix to show clusters
    clustered_corr = sns.clustermap(correlation, cmap="YlGnBu", row_cluster=True, col_cluster=True)

    # Get the sorted correlation matrix
    sorted_corr = correlation.iloc[clustered_corr.dendrogram_row.reordered_ind, clustered_corr.dendrogram_col.reordered_ind]
    return sorted_corr

#
# @app.on_event("startup")
# async def startup():


@app.get("/correlations")
async def correlations(request: Request, db: Session = Depends(get_db)):
    param_symbols = request.query_params.getlist("symbols")
    # date_from = request.query_params.get("date_from")
    # date_to = request.query_params.get("date_to")
    duration = request.query_params.get("duration", 43300)

    data_frame = pd.read_sql_query(
        db.query(model.Candles).limit(500).statement,
        db.bind
    )
    corr = preprocess(data_frame)
    return json.loads(corr.to_json())


@app.get("/symbols")
async def symbols(db: Session = Depends(get_db)):
    symbols_result = db.query(model.Candles.symbol).distinct().all()
    return [column[0] for column in symbols_result]



