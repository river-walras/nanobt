from datetime import UTC, datetime
import numpy as np
from numba import njit
from nanobt.utils import preprocess_data
from nanobt.types import record_dtype, INVALID_MIN, INVALID_MAX
from nanobt.stats import LinearAssetRecord


@njit
def accelerated_backtest(
    relative_half_spread,
    skew,
    order_notional_value,
    max_notional_position,
    fee,
    tick_size,
    lot_size,
    local_ts,
    best_bid_tick,
    best_ask_tick,
    bid_fill_tick,
    ask_fill_tick,
    order_ack_ts,
    bid_fill_tick_ack,
    ask_fill_tick_ack,
    best_bid_tick_ack,
    best_ask_tick_ack,
    bid_fill_tick_after_ack,
    ask_fill_tick_after_ack,
):
    # req_bid_tick: bid order price in ticks (limit buy order with GTX) sent to the exchange, before the exchange acknowledges it.
    # req_ask_tick: ask order price in ticks (limit sell order with GTX) sent to the exchange, before the exchange acknowledges it.
    # open_bid_tick: bid order price in ticks acknowledged by the exchange, currently an open order in the market.
    # open_ask_tick: ask order price in ticks acknowledged by the exchange, currently an open order in the market.
    #
    # INVALID_MIN and INVALID_MAX indicate that there are no orders.
    #
    # Example:
    #   If req_bid_tick is INVALID_MIN and there is an open bid order,
    #   the open bid order will be canceled (if the cancel request reaches the exchange before the order is filled).
    #   When an order is filled, its price is set to INVALID_MIN or INVALID_MAX accordingly.
    req_bid_tick = open_bid_tick = INVALID_MIN
    req_ask_tick = open_ask_tick = INVALID_MAX
    # corresponding order quantities.
    open_bid_qty = req_bid_qty = 0.0
    open_ask_qty = req_ask_qty = 0.0

    # Initial state.
    balance = 0.0
    position = 0.0
    num_trades = 0
    trading_value = 0.0
    trading_volume = 0.0

    # Row index iterator
    t = 0

    # State record for stats
    rec_i = 0
    record = np.empty(len(local_ts), record_dtype)

    while True:
        # --------------------------------------------------------
        # Local bot logic at `local_ts[t]`.
        mid_tick = (best_bid_tick[t] + best_ask_tick[t]) / 2.0
        mid_px = mid_tick * tick_size

        notional_position_value = position * mid_px
        normalized_position = notional_position_value / max_notional_position

        relative_bid_depth = relative_half_spread + skew * normalized_position
        relative_ask_depth = relative_half_spread - skew * normalized_position

        req_bid_tick = min(
            np.floor(mid_tick * (1.0 - relative_bid_depth)), best_bid_tick[t]
        )
        req_ask_tick = max(
            np.ceil(mid_tick * (1.0 + relative_ask_depth)), best_ask_tick[t]
        )

        req_bid_qty = req_ask_qty = max(
            round(order_notional_value / mid_px / lot_size) * lot_size, lot_size
        )

        # If the position exceeds the risk limit (max notional position),
        # no orders shall be open in that direction.
        if normalized_position > 1:
            req_bid_tick = INVALID_MIN
        if normalized_position < -1:
            req_ask_tick = INVALID_MAX

        # --------------------------------------------------------
        # Records the current state.
        record[rec_i].timestamp = local_ts[t]
        record[rec_i].price = mid_tick * tick_size
        record[rec_i].position = position
        record[rec_i].balance = balance * tick_size
        record[rec_i].fee = trading_value * tick_size * fee
        record[rec_i].num_trades = num_trades
        record[rec_i].trading_volume = trading_volume
        record[rec_i].trading_value = trading_value * tick_size

        rec_i += 1

        # --------------------------------------------------------
        # Processes the exchange-side logic (order fill logic).

        # If any of the requested order prices differ from the open order's price,
        # it is assumed that the bot sent the order request.
        # The request will be acknowledged and processed at `order_ack_ts[t]`.
        # Otherwise, check if the open order is filled.
        if req_bid_tick != open_bid_tick or req_ask_tick != open_ask_tick:
            # The current time is `order_ack_ts[t]`.
            order_ack_ts_ = order_ack_ts[t]

            # If there are open orders with valid prices,
            # checks whether they are filled before accepting the newly requested orders.
            if open_bid_tick > INVALID_MIN and open_bid_tick >= bid_fill_tick_ack[t]:
                execute_value = open_bid_tick * open_bid_qty
                balance -= execute_value
                position += open_bid_qty
                num_trades += 1
                trading_volume += open_bid_qty
                trading_value += execute_value
                # Invalidates the price because the order is filled.
                open_bid_tick = INVALID_MIN
            if open_ask_tick < INVALID_MAX and open_ask_tick <= ask_fill_tick_ack[t]:
                execute_value = open_ask_tick * open_ask_qty
                balance += execute_value
                position -= open_ask_qty
                num_trades += 1
                trading_volume += open_ask_qty
                trading_value += execute_value
                # Invalidates the price because the order is filled.
                open_ask_tick = INVALID_MAX

            # New orders are treated as GTX.
            # If the requested buy order price is greater than or equal to the best ask,
            # or the requested sell order price is less than or equal to the best bid,
            # the orders are rejected.
            # Invalidates the price if the order is rejected.
            if req_bid_tick >= best_ask_tick_ack[t]:
                req_bid_tick = INVALID_MIN
            if req_ask_tick <= best_bid_tick_ack[t]:
                req_ask_tick = INVALID_MAX

            # Updates the open orders to reflect accepted orders.
            open_bid_tick = req_bid_tick
            open_ask_tick = req_ask_tick
            open_bid_qty = req_bid_qty
            open_ask_qty = req_ask_qty

            # If there are open orders with valid prices,
            # checks whether they are filled before the next local timestamp (`local_ts[t+n]`)
            # that is greater than the current timestamp (`order_ack_ts[t]`).
            if (
                open_bid_tick > INVALID_MIN
                and open_bid_tick >= bid_fill_tick_after_ack[t]
            ):
                execute_value = open_bid_tick * open_bid_qty
                balance -= execute_value
                position += open_bid_qty
                num_trades += 1
                trading_volume += open_bid_qty
                trading_value += execute_value
                # Invalidates the price because the order is filled.
                open_bid_tick = INVALID_MIN
            if (
                open_ask_tick < INVALID_MAX
                and open_ask_tick <= ask_fill_tick_after_ack[t]
            ):
                execute_value = open_ask_tick * open_ask_qty
                balance += execute_value
                position -= open_ask_qty
                num_trades += 1
                trading_volume += open_ask_qty
                trading_value += execute_value
                # Invalidates the price because the order is filled.
                open_ask_tick = INVALID_MAX

            # The next local timestamp must be greater than the current timestamp (`order_ack_ts[t]`).
            while t < len(local_ts) and local_ts[t] < order_ack_ts_:
                t += 1
            # Breaks if no more rows remain for processing.
            if t == len(local_ts):
                break
        else:
            # Checks if the open orders are filled between two local timestamps.
            # The next row of data contains the bid fill price (in ticks) and ask fill price (in ticks)
            # for that interval (step).
            t += 1
            # Breaks if no more rows remain for processing.
            if t == len(local_ts):
                break

            # # If there are open orders with valid prices, checks if they are filled.
            if open_bid_tick > INVALID_MIN and open_bid_tick >= bid_fill_tick[t]:
                execute_value = open_bid_tick * open_bid_qty
                balance -= execute_value
                position += open_bid_qty
                num_trades += 1
                trading_volume += open_bid_qty
                trading_value += execute_value
                # Invalidates the price because the order is filled.
                open_bid_tick = INVALID_MIN
            if open_ask_tick < INVALID_MAX and open_ask_tick <= ask_fill_tick[t]:
                execute_value = open_ask_tick * open_ask_qty
                balance += execute_value
                position -= open_ask_qty
                num_trades += 1
                trading_volume += open_ask_qty
                trading_value += execute_value
                # Invalidates the price because the order is filled.
                open_ask_tick = INVALID_MAX

    return record[:rec_i]


def main():
    tick_size = 0.1
    lot_size = 0.001

    # if you do not have enough memory, use the code bellow
    # table = preprocess_data(
    #     tick_size=tick_size,
    #     interval_ns=1_000_000_000,
    #     start_date=datetime(2025, 11, 5, tzinfo=UTC),
    #     end_date=datetime(2025, 11, 6, tzinfo=UTC),
    #     trades_files=[
    #         "/share/tardis/raw/binance-futures/trades/BTCUSDT/binance-futures_trades_2025-11-05_BTCUSDT.parquet"
    #     ],
    #     book_ticker_files=[
    #         "/share/tardis/raw/binance-futures/book_ticker/BTCUSDT/binance-futures_book_ticker_2025-11-05_BTCUSDT.parquet"
    #     ],
    # )

    table = preprocess_data(
        tick_size=tick_size,
        interval_ns=1_000_000_000,
        start_date=datetime(2025, 11, 5, tzinfo=UTC),
        end_date=datetime(2025, 11, 10, tzinfo=UTC),
        base_dir="/share/tardis/raw",
        exchange="binance-futures",
        symbol="BTCUSDT"
    )

    local_ts = table["local_ts"].to_numpy()
    best_bid_tick = table["best_bid_tick"].to_numpy()
    best_ask_tick = table["best_ask_tick"].to_numpy()
    bid_fill_tick = table["bid_fill_tick"].to_numpy()
    ask_fill_tick = table["ask_fill_tick"].to_numpy()
    order_ack_ts = table["order_ack_ts"].to_numpy()
    bid_fill_tick_ack = table["bid_fill_tick_ack"].to_numpy()
    ask_fill_tick_ack = table["ask_fill_tick_ack"].to_numpy()
    best_bid_tick_ack = table["best_bid_tick_ack"].to_numpy()
    best_ask_tick_ack = table["best_ask_tick_ack"].to_numpy()
    bid_fill_tick_after_ack = table["bid_fill_tick_after_ack"].to_numpy()
    ask_fill_tick_after_ack = table["ask_fill_tick_after_ack"].to_numpy()

    relative_half_spread = 0.00025
    skew = relative_half_spread
    order_notional_value = 50000
    max_notional_position = order_notional_value * 20
    fee_per_value = -0.00005  # 0.005% rebates

    record = accelerated_backtest(
        relative_half_spread,
        skew,
        order_notional_value,
        max_notional_position,
        fee_per_value,
        tick_size,
        lot_size,
        local_ts,
        best_bid_tick,
        best_ask_tick,
        bid_fill_tick,
        ask_fill_tick,
        order_ack_ts,
        bid_fill_tick_ack,
        ask_fill_tick_ack,
        best_bid_tick_ack,
        best_ask_tick_ack,
        bid_fill_tick_after_ack,
        ask_fill_tick_after_ack,
    )

    stats = (
        LinearAssetRecord(record).resample("1s").stats(book_size=max_notional_position)
    )
    print(stats.summary())
    fig = stats.plot()
    fig.savefig("backtest_result.png")


if __name__ == "__main__":
    main()
