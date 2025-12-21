from typing import Literal, Sequence, Union
import numpy as np
import polars as pl


INVALID_MIN = 0
INVALID_MAX = np.iinfo(np.int64).max - 1

TIME_COL = Literal["exch_ts", "local_ts"]

PathLike = str
FilesArg = Union[PathLike, Sequence[PathLike]]
Kind = Literal["trades", "book_ticker"]
Format = Literal["parquet", "csv"]

book_ticker_schema = {
    "exchange": pl.String,
    "symbol": pl.String,
    "timestamp": pl.Int64,
    "local_timestamp": pl.Int64,
    "ask_amount": pl.Float64,
    "ask_price": pl.Float64,
    "bid_price": pl.Float64,
    "bid_amount": pl.Float64,
}


trade_schema = {
    "exchange": pl.String,
    "symbol": pl.String,
    "timestamp": pl.Int64,
    "local_timestamp": pl.Int64,
    "id": pl.String,
    "side": pl.String,
    "price": pl.Float64,
    "amount": pl.Float64,
}

record_dtype = np.dtype(
    [
        ('timestamp', 'i8'),
        ('price', 'f8'),
        ('position', 'f8'),
        ('balance', 'f8'),
        ('fee', 'f8'),
        ('num_trades', 'i8'),
        ('trading_volume', 'f8'),
        ('trading_value', 'f8')
    ],
    align=True
)
