from fastapi import FastAPI, Depends, Request
import pandas as pd
import numpy as np

import json
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.sql.elements import and_
import config
import utils

import model
from database import engine, SessionLocal
from sqlalchemy.orm import Session

app = FastAPI()

model.Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_symbols_data(param_symbols: [str], freq: utils.Frequency, date_from: str, date_to: str, db: Session):
    data_frame = pd.read_sql_query(
        db.query(model.Candles).filter(
            and_(
                model.Candles.symbol.in_(param_symbols),
                model.Candles.duration == freq.value,
                # model.Candles.timestamp >= date_from,
                # model.Candles.timestamp <= date_to
            )
        ).statement,
        db.bind
    )
    corr = utils.preprocess(data_frame, utils.Frequency.MONTHLY)
    return corr


@app.get("/correlations")
async def correlations(request: Request, db: Session = Depends(get_db)):
    param_symbols = request.query_params.get("symbols").split(",")
    date_from = request.query_params.get("start")
    date_to = request.query_params.get("end")
    freq = request.query_params.get("duration", "monthly")
    frequency = utils.Frequency[freq.upper()]

    corr = load_symbols_data(param_symbols, frequency, date_from, date_to, db)

    return json.loads(corr.to_json())


@app.get("/correlations/best")
async def correlations_best(request: Request, db: Session = Depends(get_db)):
    param_symbol = request.query_params.get("symbol")
    date_from = request.query_params.get("start")
    date_to = request.query_params.get("end")
    num_of_results = int(request.query_params.get("numOfResults", 15))
    freq = request.query_params.get("duration", "monthly")
    frequency = utils.Frequency[freq.upper()]

    symbol_data = pd.read_sql_query(
        db.query(model.Candles).filter(
            and_(
                model.Candles.symbol == param_symbol,
                model.Candles.duration == frequency.value,
                # model.Candles.timestamp >= date_from,
                # model.Candles.timestamp <= date_to
            )).statement,
        db.bind
    )
    symbol_data = utils.pivot_dataframe(symbol_data, frequency)
    symbol_data = utils.filter_dataframe(symbol_data)

    if symbol_data.shape[1] == 0:
        # TODO return error
        return json.loads(pd.DataFrame().to_json())

    symbol_data = symbol_data.interpolate(method='linear')

    # gets all symbols from the database
    symbols_result = db.query(model.Candles.symbol).distinct().all()
    symbols_result = [column[0] for column in symbols_result]

    index = 0
    batch_size = 1000

    best_symbols = np.zeros(batch_size + num_of_results, dtype=[('symbol', 'U30'), ('corr', float)])

    best_symbols[0] = (param_symbol, symbol_data[param_symbol].corr(symbol_data[param_symbol]))
    print(best_symbols[0])

    while index < len(symbols_result):
        end_index = index + batch_size > len(symbols_result) and len(symbols_result) or index + batch_size
        symbols_batch = symbols_result[index:end_index]
        batch_data_frame = pd.read_sql_query(
            db.query(model.Candles.open, model.Candles.timestamp, model.Candles.symbol).where(
                    and_(
                        model.Candles.symbol.in_(symbols_batch),
                        model.Candles.duration == frequency.value
                    )
            ).statement,
            db.bind
        )
        batch_data = utils.pivot_dataframe(batch_data_frame, utils.Frequency.MONTHLY)
        batch_data = utils.filter_dataframe(batch_data)
        batch_data = batch_data.interpolate(method='linear')

        batch_data[param_symbol] = symbol_data[param_symbol]
        for i, series_symbol in enumerate(batch_data):
            if series_symbol == param_symbol:
                continue
            corr = batch_data[param_symbol].corr(batch_data[series_symbol])
            best_symbols[num_of_results + i] = (series_symbol, corr)

        best_symbols = utils.sort_best_correlations(best_symbols)
        index += batch_size

    return_symbols = best_symbols[0:num_of_results]['symbol']
    corr = load_symbols_data(return_symbols, frequency, date_from, date_to, db)
    return json.loads(corr.to_json())


@app.get("/symbols")
async def symbols(db: Session = Depends(get_db)):
    symbols_result = db.query(model.Candles.symbol).distinct().all()
    return [column[0] for column in symbols_result]



