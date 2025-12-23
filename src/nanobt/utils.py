from datetime import datetime, timezone
from typing import Literal, Optional

import polars as pl
import numpy as np
import numba as nb
from numba import njit
from numba.experimental import jitclass

from .types import (
    TIME_COL,
    INVALID_MAX,
    INVALID_MIN,
    FilesArg,
    Kind,
    Format,
)


@jitclass
class Clock:
    timestamp: nb.int64[:]  # type: ignore
    rn: nb.int64  # type: ignore
    ts: nb.int64  # type: ignore

    def __init__(self, timestamp, rn):
        self.timestamp = timestamp
        self.rn = rn
        if self.rn >= len(self.timestamp):
            self.ts = INVALID_MAX
        else:
            self.ts = self.timestamp[self.rn]

    def next(self):
        if self.rn == len(self.timestamp) - 1:
            self.ts = INVALID_MAX
        else:
            self.rn += 1
            self.ts = self.timestamp[self.rn]


@njit
def select_event(timestamps):
    # Finds the earliest timestamped event to process first.
    earliest_ts = INVALID_MAX
    ev = -1
    for i in range(len(timestamps)):
        if timestamps[i] < earliest_ts:
            earliest_ts = timestamps[i]
            ev = i
    return ev


def datetime_to_ns(dt: datetime) -> int:
    """
    Convert datetime to int64 nanoseconds since epoch.
    - naive datetime: treated as UTC
    - tz-aware: converted to UTC
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def load_data(
    files: FilesArg,
    *,
    start: Optional[datetime] = None,  # datetime → ns
    end: Optional[datetime] = None,  # datetime → ns
    kind: Kind,  # Must be explicitly specified
    format: Format = "parquet",  # File format
    to_numpy: bool = True,
    time_col: TIME_COL = "exch_ts",
    sort_by: Optional[TIME_COL] = "exch_ts",  # None = don't sort
):
    """
    Load ONE kind of data using Polars scan_parquet or scan_csv.

    - files: str | list[str]
        - Can be a glob pattern (scan_parquet/scan_csv handles it)
        - Can be list[path]
    - kind: "trades" | "book_ticker" (must specify)
    - format: "parquet" | "csv" (default parquet)
    """
    start_ns = datetime_to_ns(start) if start is not None else None
    end_ns = datetime_to_ns(end) if end is not None else None

    source = files  # 原样交给 scan_parquet/scan_csv（支持 glob / list）

    # 根据 format 选择 scan 函数
    if format == "parquet":
        scan_fn = pl.scan_parquet
    elif format == "csv":
        scan_fn = pl.scan_csv
    else:
        raise ValueError(f"Unsupported format: {format}")

    if kind == "trades":
        lf = scan_fn(source).select(
            pl.when(pl.col("side") == "buy")
            .then(1)
            .otherwise(-1)
            .cast(pl.Int8)
            .alias("side"),
            (pl.col("timestamp") * 1000).cast(pl.Int64, strict=True).alias("exch_ts"),
            (pl.col("local_timestamp") * 1000)
            .cast(pl.Int64, strict=True)
            .alias("local_ts"),
            pl.col("price").cast(pl.Float64, strict=True).alias("px"),
            pl.col("amount").cast(pl.Float64, strict=True).alias("qty"),
        )

    elif kind == "book_ticker":
        lf = scan_fn(source).select(
            (pl.col("timestamp") * 1000).cast(pl.Int64, strict=True).alias("exch_ts"),
            (pl.col("local_timestamp") * 1000)
            .cast(pl.Int64, strict=True)
            .alias("local_ts"),
            pl.col("ask_price").cast(pl.Float64, strict=True).alias("ask_px"),
            pl.col("ask_amount").cast(pl.Float64, strict=True).alias("ask_qty"),
            pl.col("bid_price").cast(pl.Float64, strict=True).alias("bid_px"),
            pl.col("bid_amount").cast(pl.Float64, strict=True).alias("bid_qty"),
        )
    else:
        raise ValueError(f"Unsupported kind: {kind}")

    if start_ns is not None:
        lf = lf.filter(pl.col(time_col) >= start_ns)
    if end_ns is not None:
        lf = lf.filter(pl.col(time_col) < end_ns)

    df = lf.collect()

    if sort_by is not None and sort_by in df.columns:
        df = df.sort(sort_by)

    if not to_numpy:
        return df["exch_ts"], df["local_ts"], df

    return (
        df["exch_ts"].to_numpy(),
        df["local_ts"].to_numpy(),
        df.to_numpy(structured=True),
    )


@njit
def ack_order(
    tick_size,
    order_ack_ts,
    next_local_ts,
    book_ticker_rn,
    book_ticker_exch_ts,
    book_ticker,
    trades_rn,
    trades_exch_ts,
    trades,
):
    # This function finds the fill prices around order_ack_ts, as well as the best bid and best ask at order_ack_ts.
    # - Fill prices between local_ts[t] and order_ack_ts[t]
    # - Fill prices between order_ack_ts[t] and local_ts[t + n], where local_ts[t + n] > order_ack_ts[t]

    # Initializes the values from the last best bid and best ask.
    best_bid_tick = round(book_ticker[book_ticker_rn - 1].bid_px / tick_size)
    ask_fill_tick = ask_fill_tick_ack = best_bid_tick_ack = high_best_bid_tick = (
        best_bid_tick
    )
    best_ask_tick = round(book_ticker[book_ticker_rn - 1].ask_px / tick_size)
    bid_fill_tick = bid_fill_tick_ack = best_ask_tick_ack = low_best_ask_tick = (
        best_ask_tick
    )

    high_buy_tick = INVALID_MIN
    low_sell_tick = INVALID_MAX

    book_ticker_exch_clock = Clock(book_ticker_exch_ts, book_ticker_rn)
    trades_exch_clock = Clock(trades_exch_ts, trades_rn)

    while True:
        ev = select_event(
            np.asarray(
                [
                    book_ticker_exch_clock.ts,
                    trades_exch_clock.ts,
                    order_ack_ts,
                    next_local_ts,
                ]
            )
        )

        if ev == -1:
            raise ValueError
        elif ev == 0:
            # Find best bid and best ask for [local_ts[t], order_ack_ts[t]] and [order_ack_ts[t], local_ts[t + n]] intervals
            best_bid_tick = round(
                book_ticker[book_ticker_exch_clock.rn].bid_px / tick_size
            )
            best_ask_tick = round(
                book_ticker[book_ticker_exch_clock.rn].ask_px / tick_size
            )

            if best_bid_tick > high_best_bid_tick:
                high_best_bid_tick = best_bid_tick
            if best_ask_tick < low_best_ask_tick:
                low_best_ask_tick = best_ask_tick

            book_ticker_exch_clock.next()
        elif ev == 1:
            # Find trade prices for [local_ts[t], order_ack_ts[t]] and [order_ack_ts[t], local_ts[t + n]] intervals
            side = trades[trades_exch_clock.rn].side
            px_tick = round(trades[trades_exch_clock.rn].px / tick_size)

            if side == 1 and px_tick > high_buy_tick:
                high_buy_tick = px_tick
            elif side == -1 and px_tick < low_sell_tick:
                low_sell_tick = px_tick

            trades_exch_clock.next()
        elif ev == 2:
            # Describes the [local_ts[t], order_ack_ts[t]] interval case, i.e., new order hasn't reached exchange yet
            # An order request is acknowledged by the exchange at order_ack_ts[t].
            bid_fill_tick_ack = min(low_sell_tick + 1, low_best_ask_tick)
            ask_fill_tick_ack = max(high_buy_tick - 1, high_best_bid_tick)
            best_bid_tick_ack = best_bid_tick  # Record the best_bid_tick when order reaches exchange
            best_ask_tick_ack = best_ask_tick  # Record the best_ask_tick when order reaches exchange

            # Reinitialize for entering the [order_ack_ts[t], local_ts[t + n]] interval calculation
            # Note: this is not necessarily t+1, but t+n, since order latency may span multiple local time steps
            high_buy_tick = INVALID_MIN
            high_best_bid_tick = best_bid_tick
            low_sell_tick = INVALID_MAX
            low_best_ask_tick = best_ask_tick

            order_ack_ts = INVALID_MAX
        elif ev == 3:
            # Describes the [order_ack_ts[t], local_ts[t + n]] interval case, i.e., new order has reached exchange
            # at local_ts[t + n] > order_ack_ts[t]
            bid_fill_tick = min(low_sell_tick + 1, low_best_ask_tick)
            ask_fill_tick = max(high_buy_tick - 1, high_best_bid_tick)
            break

    return (
        bid_fill_tick_ack,
        ask_fill_tick_ack,
        best_bid_tick_ack,
        best_ask_tick_ack,
        bid_fill_tick,
        ask_fill_tick,
    )


@njit
def _preprocess_data(
    tick_size,
    end_ts,
    local_ts,
    book_ticker_exch_ts,
    book_ticker_local_ts,
    book_ticker,
    trades_exch_ts,
    trades,
    fixed_entry_lat_ns,
):
    """preprocess_data
    Preprocesses the raw market data to generate necessary information for backtesting.

    Parameters
    ----------
    tick_size : float
        The minimum price increment for the trading instrument.
    end_ts : int
        The end timestamp for the preprocessing.
    local_ts : np.ndarray
        An array of local timestamps at which the local system observes the market.
    book_ticker_exch_ts : np.ndarray
        An array of exchange timestamps for the book ticker data.
    book_ticker_local_ts : np.ndarray
        An array of local timestamps for the book ticker data.
    book_ticker : np.ndarray
        An array of book ticker data records.
    trades_exch_ts : np.ndarray
        An array of exchange timestamps for the trades data.
    trades : np.ndarray
        An array of trade data records.
    fixed_entry_lat_ns : int
        The fixed latency in nanoseconds for order entry from local system to exchange.
    """
    # W0 = local_ts[t-1] ~ local_ts[t] # Whether legacy orders are filled
    # W1 = local_ts[t] ~ order_ack_ts[t] # For modify/cancel orders, whether the order is filled
    # W2 = order_ack_ts[t] ~ local_ts[t + n] where local_ts[t + n] > order_ack_ts[t] # Whether new orders are filled in this window

    # out_bid_fill_tick, out_ask_fill_tick: Used to determine if orders in local_ts[t-1]~local_ts[t] interval are filled, i.e., legacy order fill status
    # out_bid_fill_tick_ack, out_ask_fill_tick_ack: Used to determine if orders in local_ts[t]~order_ack_ts[t] interval are filled, i.e., if order is filled before reaching exchange after sending modify/cancel
    # out_bid_fill_tick_after_ack, out_ask_fill_tick_after_ack: Used to determine if orders in order_ack_ts[t]~local_ts[t + n] interval are filled, i.e., new order fill status

    # Preprocessed data
    # All prices are in ticks to avoid additional operations to prevent floating-point comparison errors.

    # W0 = [local_ts[t-1], local_ts[t]]
    #   -> Whether legacy orders (already open orders on the exchange) are filled in this previous segment

    # W1 = [local_ts[t], order_ack_ts[t]]
    #   -> After local system sends cancel/modify request at t, whether the legacy order will be
    #      filled first before the request reaches exchange (cancel/modify timeout)

    # W2 = [order_ack_ts[t], next_local_ts]  where next_local_ts = first local_ts >= order_ack_ts[t]
    #   -> After newly issued orders reach the exchange, whether they are filled before the next local observation point

    out_t = 0
    out_size = len(local_ts)
    # timestamp at local := local_ts[t]
    out_local_ts = np.empty(out_size, np.int64)
    # best bid at local at local_ts[t]
    out_best_bid_tick = np.empty(out_size, np.int64)
    # best bid qty at local at local_ts[t]
    out_best_bid_qty = np.empty(out_size, np.float64)
    # best ask at local at local_ts[t]
    out_best_ask_tick = np.empty(out_size, np.int64)
    # out best ask qty at local at local_ts[t]
    out_best_ask_qty = np.empty(out_size, np.float64)
    # bid fill price in ticks for the interval local_ts[t - 1] ~ local_ts[t]
    # any open buy orders during this interval with a price greater than or equal to this price are considered filled.
    out_bid_fill_tick = np.empty(out_size, np.int64)
    # ask fill price in ticks for the interval local_ts[t - 1] ~ local_ts[t]
    # any open sell orders during this interval with a price less than or equal to this price are considered filled.
    out_ask_fill_tick = np.empty(out_size, np.int64)
    # order acknowledgment timestamp at the exchange, when an order is sent at local_ts[t], is defined as order_ack_ts[t].
    out_order_ack_ts = np.empty(out_size, np.int64)
    # bid fill price in ticks for the interval local_ts[t] ~ order_ack_ts[t]
    # any open buy orders during this interval with a price greater than or equal to this price are considered filled.
    out_bid_fill_tick_ack = np.empty(out_size, np.int64)
    # ask fill price in ticks for the interval local_ts[t] ~ order_ack_ts[t]
    # any open sell orders during this interval with a price less than or equal to this price are considered filled.
    out_ask_fill_tick_ack = np.empty(out_size, np.int64)
    # best bid at the exchange at order_ack_ts[t]
    # used to determine whether the order should be accepted (limit or market) or rejected (GTX).
    out_best_bid_tick_ack = np.empty(out_size, np.int64)
    # best ask at the exchange at order_ack_ts[t]
    # used to determine whether the order should be accepted (limit or market) or rejected (GTX).
    out_best_ask_tick_ack = np.empty(out_size, np.int64)
    # bid fill price in ticks for the interval order_ack_ts[t] ~ local_ts[t + n] where local_ts[t + n] > order_ack_ts[t].
    # any open buy orders during this interval with a price greater than or equal to this price are considered filled.
    out_bid_fill_tick_after_ack = np.empty(out_size, np.int64)
    # ask fill price in ticks for the interval order_ack_ts[t] ~ local_ts[t + n] where local_ts[t + n] > order_ack_ts[t].
    # any open sell orders during this interval with a price less than or equal to this price are considered filled.
    out_ask_fill_tick_after_ack = np.empty(out_size, np.int64)

    local_best_bid_tick = exch_best_bid_tick = INVALID_MIN
    local_best_ask_tick = exch_best_ask_tick = INVALID_MAX
    local_best_bid_qty = 0.0
    local_best_ask_qty = 0.0
    high_buy_tick = high_best_bid_tick = INVALID_MIN
    low_sell_tick = low_best_ask_tick = INVALID_MAX

    # Initializes the clocks
    # todo: For better accuracy, also need to combine best bid and ask from Level-2 market depth data
    #       with best bid and ask from book ticker.
    book_ticker_exch_clock = Clock(book_ticker_exch_ts, 0)  # rn = 0
    book_ticker_local_clock = Clock(book_ticker_local_ts, 0)  # rn = 0
    trades_exch_clock = Clock(trades_exch_ts, 0)  # rn = 0
    local_clock = Clock(local_ts, 0)  # rn = 0

    last_local_ts = local_ts[len(local_ts) - 1]

    while local_clock.ts <= end_ts:
        # Selects the event to process.
        ev = select_event(
            np.asarray(
                [
                    book_ticker_exch_clock.ts,
                    book_ticker_local_clock.ts,
                    trades_exch_clock.ts,
                    local_clock.ts,
                ]
            )
        )

        if ev == -1:
            # Should not happen.
            raise ValueError
        elif ev == 0:
            # Updates the current exchange best bid and best ask.
            exch_best_bid_tick = round(
                book_ticker[book_ticker_exch_clock.rn].bid_px / tick_size
            )
            exch_best_ask_tick = round(
                book_ticker[book_ticker_exch_clock.rn].ask_px / tick_size
            )

            # Updates the highest and lowest best bid and best ask at the exchange.
            if exch_best_bid_tick > high_best_bid_tick:
                high_best_bid_tick = exch_best_bid_tick
            if exch_best_ask_tick < low_best_ask_tick:
                low_best_ask_tick = exch_best_ask_tick

            book_ticker_exch_clock.next()
        elif ev == 1:
            # Updates the current local best bid and best ask.
            local_best_bid_tick = round(
                book_ticker[book_ticker_local_clock.rn].bid_px / tick_size
            )
            local_best_ask_tick = round(
                book_ticker[book_ticker_local_clock.rn].ask_px / tick_size
            )
            local_best_bid_qty = book_ticker[book_ticker_local_clock.rn].bid_qty
            local_best_ask_qty = book_ticker[book_ticker_local_clock.rn].ask_qty

            book_ticker_local_clock.next()
        elif ev == 2:
            side = trades[trades_exch_clock.rn].side
            px_tick = round(trades[trades_exch_clock.rn].px / tick_size)

            # Updates the highest and lowest trade at the exchange.
            if side == 1 and px_tick > high_buy_tick:
                high_buy_tick = px_tick
            elif side == -1 and px_tick < low_sell_tick:
                low_sell_tick = px_tick

            trades_exch_clock.next()
        elif ev == 3:
            # W0 = local_ts[t-1] ~ local_ts[t]
            # Records the fill prices in ticks at the exchange between local_ts[t - 1] and local_ts[t].
            out_bid_fill_tick[out_t] = min(low_sell_tick + 1, low_best_ask_tick)
            out_ask_fill_tick[out_t] = max(high_buy_tick - 1, high_best_bid_tick)

            high_buy_tick = INVALID_MIN  # If no trade updates, set to minimum value
            high_best_bid_tick = exch_best_bid_tick  # Reset to current exch best bid, as next interval may have no book updates
            low_sell_tick = INVALID_MAX
            low_best_ask_tick = exch_best_ask_tick

            # Records the current local state at local_ts[t].
            out_local_ts[out_t] = local_clock.ts
            out_best_bid_tick[out_t] = local_best_bid_tick
            out_best_bid_qty[out_t] = local_best_bid_qty
            out_best_ask_tick[out_t] = local_best_ask_tick
            out_best_ask_qty[out_t] = local_best_ask_qty

            # Order acknowledgement timestamp when the exchange receives the order request.
            # order_entry_latency = order_latency.entry(local_clock.ts)
            order_ack_ts = local_clock.ts + fixed_entry_lat_ns
            if order_ack_ts > last_local_ts:
                order_ack_ts = last_local_ts

            # The next local timestamp after the exchange acknowledges the order request.
            # Finds the first local_ts >= order_ack_ts
            next_local_clock = Clock(local_ts, local_clock.rn)
            while next_local_clock.ts < order_ack_ts:
                next_local_clock.next()
            next_local_ts = next_local_clock.ts

            # Computes the fill prices around the order acknowledgment timestamp at the exchange
            # and finds the best bid and best ask at the time of acknowledgment.
            (
                bid_fill_tick_ack,
                ask_fill_tick_ack,
                best_bid_tick_ack,
                best_ask_tick_ack,
                bid_fill_tick,
                ask_fill_tick,
            ) = ack_order(
                tick_size,
                order_ack_ts,
                next_local_ts,
                book_ticker_exch_clock.rn + 1,
                book_ticker_exch_ts,
                book_ticker,
                trades_exch_clock.rn + 1,
                trades_exch_ts,
                trades,
            )

            # Records the values related to the order acknowledgment.
            out_order_ack_ts[out_t] = order_ack_ts
            out_bid_fill_tick_ack[out_t] = bid_fill_tick_ack
            out_ask_fill_tick_ack[out_t] = ask_fill_tick_ack
            out_best_bid_tick_ack[out_t] = best_bid_tick_ack
            out_best_ask_tick_ack[out_t] = best_ask_tick_ack
            out_bid_fill_tick_after_ack[out_t] = bid_fill_tick
            out_ask_fill_tick_after_ack[out_t] = ask_fill_tick

            out_t += 1

            local_clock.next()

    return (
        out_local_ts[:out_t],
        out_best_bid_tick[:out_t],
        out_best_bid_qty[:out_t],
        out_best_ask_tick[:out_t],
        out_best_ask_qty[:out_t],
        out_bid_fill_tick[:out_t],
        out_ask_fill_tick[:out_t],
        out_order_ack_ts[:out_t],
        out_bid_fill_tick_ack[:out_t],
        out_ask_fill_tick_ack[:out_t],
        out_best_bid_tick_ack[:out_t],
        out_best_ask_tick_ack[:out_t],
        out_bid_fill_tick_after_ack[:out_t],
        out_ask_fill_tick_after_ack[:out_t],
    )


def generate_tardis_file_paths(
    start_date: datetime,
    end_date: datetime,
    data_type: Literal["trades", "book_ticker"],
    base_dir: str = "/share/tardis/raw/",
    exchange: str = "binance-futures",
    symbol: str = "BTCUSDT",
    format: Format = "parquet",
) -> list[str]:
    """
    Generate file paths for tardis data files within a date range.

    Parameters
    ----------
    start_date : datetime
        Start date (inclusive)
    end_date : datetime
        End date (exclusive)
    data_type : Literal["trades", "book_ticker"]
        Type of data files to generate paths for
    base_dir : str
        Base directory for tardis data
    exchange : str
        Exchange name
    symbol : str
        Trading symbol

    Returns
    -------
    list[str]
        List of file paths for the date range
    """
    from pathlib import Path
    from datetime import timedelta

    if format not in ("parquet", "csv"):
        raise ValueError(f"Unsupported format: {format}")

    if format == "csv":
        _format = "csv.gz"  # Tardis CSV files are gzipped
    else:
        _format = format

    file_paths = []
    current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date_normalized = end_date.replace(hour=0, minute=0, second=0, microsecond=0)

    while current_date < end_date_normalized:
        date_str = current_date.strftime("%Y-%m-%d")
        file_name = f"{exchange}_{data_type}_{date_str}_{symbol}.{_format}"
        file_path = str(Path(base_dir) / exchange / data_type / symbol / file_name)
        file_paths.append(file_path)
        current_date += timedelta(days=1)

    return file_paths


def preprocess_data(
    tick_size: float,
    interval_ns: int,
    start_date: datetime,
    end_date: datetime,
    format: Format = "parquet",
    trades_files: FilesArg | None = None,
    book_ticker_files: FilesArg | None = None,
    base_dir: str | None = None,
    exchange: str | None = None,
    symbol: str | None = None,
    fixed_entry_lat_ns: int = 10_000_000, # 10 ms
) -> pl.DataFrame:
    """
    Preprocess market data for backtesting.

    Parameters
    ----------
    start_date : datetime
        Start date for data processing
    end_date : datetime
        End date for data processing
    tick_size : float
        The minimum price increment for the trading instrument
    lot_size : float
        The minimum quantity increment for the trading instrument
    base_dir : str
        Base directory for tardis data files
    exchange : str
        Exchange name
    symbol : str
        Trading symbol
    interval_ns : int
        Interval in nanoseconds for running the backtest
    fixed_entry_lat_ns : int
        Fixed entry latency in nanoseconds

    Returns
    -------
    pl.DataFrame
        Preprocessed data with columns for local_ts, best_bid_tick, best_ask_tick, etc.
    """
    if not (trades_files and book_ticker_files):
        if not (base_dir and exchange and symbol):
            raise ValueError(
                "Either provide trades_files and book_ticker_files, or base_dir, exchange, and symbol."
            )

    # Generate file paths
    trades_files = trades_files or generate_tardis_file_paths(
        start_date, end_date, "trades", base_dir, exchange, symbol, format # type: ignore
    )
    book_ticker_files = book_ticker_files or generate_tardis_file_paths(
        start_date, end_date, "book_ticker", base_dir, exchange, symbol, format # type: ignore
    )

    # Load data
    trade_exch_ts, _, trades = load_data(
        trades_files,
        to_numpy=True,
        kind="trades",
        format=format,
    )
    book_ticker_exch_ts, book_ticker_local_ts, book_ticker = load_data(
        book_ticker_files,
        to_numpy=True,
        kind="book_ticker",
        format=format,
    )

    # Generate timestamps
    start_ts = int(start_date.timestamp() * 1_000_000_000) + interval_ns
    end_ts = int(end_date.timestamp() * 1_000_000_000)

    # In the final interval, to compute fill prices after order acknowledgment,
    # a small buffer beyond `end_ts` is necessary.
    local_ts = np.arange(start_ts, end_ts + interval_ns, interval_ns)

    # Preprocess data
    out = _preprocess_data(
        tick_size,
        end_ts,
        local_ts,
        book_ticker_exch_ts,
        book_ticker_local_ts,
        book_ticker,
        trade_exch_ts,
        trades,
        fixed_entry_lat_ns=fixed_entry_lat_ns,
    )

    # Create DataFrame
    table = pl.DataFrame(
        {
            "local_ts": out[0],
            "best_bid_tick": out[1],
            "best_bid_qty": out[2],
            "best_ask_tick": out[3],
            "best_ask_qty": out[4],
            "bid_fill_tick": out[5],
            "ask_fill_tick": out[6],
            "order_ack_ts": out[7],
            "bid_fill_tick_ack": out[8],
            "ask_fill_tick_ack": out[9],
            "best_bid_tick_ack": out[10],
            "best_ask_tick_ack": out[11],
            "bid_fill_tick_after_ack": out[12],
            "ask_fill_tick_after_ack": out[13],
        }
    )

    return table
