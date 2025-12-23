# NanoBT - Nanosecond-Level Accelerated Backtesting Framework

## Core Design Principles

### Acceleration Method: Pre-computed Fill Prices

Traditional backtesting requires tick-by-tick replay of market depth data and simulation of order book queue positions, while NanoBT uses a **pre-computed fill price** approach:

- **Buy Order Fill Condition**: Order price ≥ minimum ask price in interval or > minimum active sell execution price
- **Sell Order Fill Condition**: Order price ≤ maximum bid price in interval or < maximum active buy execution price

**Trade-off**: Queue position is not modeled; orders are either fully filled or not filled (no partial fills). In high-liquidity markets (such as mainstream cryptocurrency futures), this simplification is acceptable for most strategies' accuracy.

---

## Detailed Explanation of Matching Logic

### Time Axis Segmentation: Three Key Windows

The backtesting system divides the time axis into three windows, handling different states of order fill determination:

```
Time axis:
    local_ts[t-1]      local_ts[t]         order_ack_ts[t]      local_ts[t+n]
        |                  |                     |                   |
        |<------ W0 ------>|<------- W1 -------->|<------ W2 ------->|
    legacy order fill    cancel/modify timeout    new order fill
```

#### **W0: Legacy Order Window** `[local_ts[t-1], local_ts[t]]`
Determines if open orders submitted in the previous strategy execution are filled before the current timestamp.

- **Output Data**:
  - `bid_fill_tick`: Buy order fill price (in ticks); any open order at price ≥ this value is considered filled
  - `ask_fill_tick`: Sell order fill price (in ticks); any open order at price ≤ this value is considered filled

- **Calculation Method** (see [utils.py:417-420](src/nanobt/utils.py#L417-L420)):
  ```python
  # Lowest active sell execution price + 1 tick, minimum with lowest ask
  bid_fill_tick = min(low_sell_tick + 1, low_best_ask_tick)
  # Highest active buy execution price - 1 tick, maximum with highest bid
  ask_fill_tick = max(high_buy_tick - 1, high_best_bid_tick)
  ```

#### **W1: Order Modification Latency Window** `[local_ts[t], order_ack_ts[t]]`
After the local system sends a cancel/modify request at time `t`, determines if the original order will be **filled before the request arrives** at the exchange (cancel/modify timeout).

- **Output Data**:
  - `bid_fill_tick_ack`: Buy order latency fill price (in ticks)
  - `ask_fill_tick_ack`: Sell order latency fill price (in ticks)
  - `best_bid_tick_ack`: Best bid price when the exchange receives the order (for GTX rejection)
  - `best_ask_tick_ack`: Best ask price when the exchange receives the order (for GTX rejection)

- **Use Case** (see [main.py:111-128](main.py#L111-L128)):
  When the strategy attempts to cancel or modify an order, determine if it has already been filled before reaching the exchange:
  ```python
  # Check if the open order is filled before reaching the exchange
  if open_bid_tick > INVALID_MIN and open_bid_tick >= bid_fill_tick_ack[t]:
      # Buy order filled: deduct balance, increase position
      balance -= open_bid_tick * open_bid_qty
      position += open_bid_qty
  ```

#### **W2: New Order Window** `[order_ack_ts[t], local_ts[t+n]]`
After the exchange confirms the order, determines if the new order will be filled before the next strategy observation point.

- **Output Data**:
  - `bid_fill_tick_after_ack`: New buy order fill price (in ticks)
  - `ask_fill_tick_after_ack`: New sell order fill price (in ticks)

- **Note**: `local_ts[t+n]` here is not necessarily `t+1`, as order latency may span multiple local time steps.

---

### Latency Simulation: Order Acknowledgment Timestamp

The system simulates the **one-way latency** (Entry Latency) from local system to exchange via `order_ack_ts`:

```python
# utils.py:434
order_ack_ts = local_ts[t] + fixed_entry_lat_ns
```

- **Default Latency**: 10ms (`fixed_entry_lat_ns = 10_000_000` nanoseconds)
- **Future Improvements**: Can be extended to interpolation-based latency models from historical data (reference HftBacktest's `IntpOrderLatency`)

**Important Limitation**: Order state changes (acknowledgment/cancellation/fill/position update) take effect **locally immediately**, not included in order response latency. This avoids the complexity of maintaining separate local and exchange states.

---

### GTX Order Type: Prevent Immediate Execution

The system treats all limit orders as **GTX (Good-Till-Crossing)** orders, meaning "post-only":

```python
# main.py:130-138
# If buy order price >= ask, or sell order price <= bid, order is rejected
if req_bid_tick >= best_ask_tick_ack[t]:
    req_bid_tick = INVALID_MIN  # Reject order
if req_ask_tick <= best_bid_tick_ack[t]:
    req_ask_tick = INVALID_MAX  # Reject order
```

这避免了策略因"越过价差"而吃掉对手单，确保**只做 Maker 不做 Taker**。

---

### Order State Management: Four Price Variables

The system uses four variables to track the order lifecycle (see [main.py:30-45](main.py#L30-L45)):

```python
# Unacknowledged orders (locally sent, exchange not confirmed)
req_bid_tick = INVALID_MIN  # Pending buy order price (in ticks)
req_ask_tick = INVALID_MAX  # Pending sell order price (in ticks)

# Acknowledged orders (exchange accepted, order open in market)
open_bid_tick = INVALID_MIN  # Buy order price in market (in ticks)
open_ask_tick = INVALID_MAX  # Sell order price in market (in ticks)
```

- **INVALID_MIN / INVALID_MAX**: Special marker values indicating "no order" or "order filled"
  - `INVALID_MIN = 0`: Used for buy orders
  - `INVALID_MAX = 2^63 - 2`: Used for sell orders

**Order Cancellation Logic** (see [main.py:38-40](main.py#L38-L40)):
```python
# If req_bid_tick is INVALID_MIN and there is an open buy order, send cancel request
# (If cancel request reaches exchange before order fills, order will be canceled)
if req_bid_tick == INVALID_MIN and open_bid_tick > INVALID_MIN:
    # Exchange will process cancellation at order_ack_ts[t], provided order hasn't filled in W1 window
```

---

## Example Strategy: Skew Market Making Strategy

The built-in example strategy (see [main.py:64-84](main.py#L64-L84)) implements a **position-skewed dual-sided market-making strategy**:

### Core Concept
1. Place symmetric market making orders above and below the mid price
2. Dynamically adjust buy/sell order depth (skew) based on current position
3. Stop one-sided order placement when reaching risk limits

### Parameter Configuration
```python
relative_half_spread = 0.00025  # Half spread: 0.025% (base distance from mid price)
skew = 0.00025                  # Skew coefficient: adjust order distance based on position
order_notional_value = 50000    # Order notional value per order: $50,000
max_notional_position = 1000000 # Max position notional value: $1,000,000 (20x leverage)
```

### Order Placement Logic
```python
mid_tick = (best_bid_tick[t] + best_ask_tick[t]) / 2.0  # Mid price
normalized_position = (position * mid_px) / max_notional_position  # Normalized position ∈ [-1, 1]

# Adjust buy/sell depth based on position skew
relative_bid_depth = relative_half_spread + skew * normalized_position
relative_ask_depth = relative_half_spread - skew * normalized_position

# Calculate order prices (must not cross spread)
req_bid_tick = min(floor(mid_tick * (1 - relative_bid_depth)), best_bid_tick[t])
req_ask_tick = max(ceil(mid_tick * (1 + relative_ask_depth)), best_ask_tick[t])

# Risk control: stop one-sided orders at position limits
if normalized_position > 1:  # Long position over limit
    req_bid_tick = INVALID_MIN  # Stop buying
if normalized_position < -1:  # Short position over limit
    req_ask_tick = INVALID_MAX  # Stop selling
```

**Skew Effect**:
- When long: buy order depth increases (less aggressive buy), sell order depth decreases (aggressive sell to close)
- When short: sell order depth increases (less aggressive sell), buy order depth decreases (aggressive buy to close)

---

## Data Preprocessing Pipeline

### Input Data Requirements

The system requires two types of raw market data (Tardis format):

1. **Trade-by-trade data** (`trades`)
   ```python
   # Required fields: timestamp, local_timestamp, side, price, amount
   side: 1 (buy) / -1 (sell)  # Active buy/sell direction
   ```

2. **Best bid/ask data** (`book_ticker`)
   ```python
   # Required fields: timestamp, local_timestamp, bid_price, bid_amount, ask_price, ask_amount
   ```

### Preprocessing Steps

[preprocess_data](src/nanobt/utils.py#L552-L654) function's core workflow:

1. **Load data** (supports Parquet/CSV format)
   ```python
   trades = load_data(files, kind="trades")
   book_ticker = load_data(files, kind="book_ticker")
   ```

2. **Generate local timestamp sequence**
   ```python
   # Generate strategy execution time points at fixed intervals (e.g., 1 second)
   local_ts = np.arange(start_ts, end_ts, interval_ns)
   ```

3. **Event-driven iteration** (see [utils.py:363-477](src/nanobt/utils.py#L363-L477))

   Use **Clock** objects to manage four event streams:
   - `book_ticker_exch_clock`: Best bid/ask updates at exchange timestamps
   - `book_ticker_local_clock`: Best bid/ask updates at local timestamps
   - `trades_exch_clock`: Trade-by-trade data
   - `local_clock`: Strategy execution time points

   **Each iteration** processes the earliest timestamp event:
   ```python
   ev = select_event([
       book_ticker_exch_clock.ts,
       book_ticker_local_clock.ts,
       trades_exch_clock.ts,
       local_clock.ts
   ])
   ```

4. **Calculate fill prices**

   At each `local_ts[t]`, call `ack_order` function to compute fill prices for three windows:
   ```python
   # W0: Legacy orders
   bid_fill_tick = min(low_sell_tick + 1, low_best_ask_tick)
   ask_fill_tick = max(high_buy_tick - 1, high_best_bid_tick)

   # W1 & W2: Computed via ack_order function
   (bid_fill_tick_ack, ask_fill_tick_ack,
    best_bid_tick_ack, best_ask_tick_ack,
    bid_fill_tick_after_ack, ask_fill_tick_after_ack) = ack_order(...)
   ```

5. **Output preprocessed table**

   Returns Polars DataFrame with one row per strategy time point:
   ```python
   {
       'local_ts',               # Local timestamp
       'best_bid_tick',          # Local best bid price
       'best_ask_tick',          # Local best ask price
       'bid_fill_tick',          # W0 buy order fill price
       'ask_fill_tick',          # W0 sell order fill price
       'order_ack_ts',           # Order acknowledgment timestamp
       'bid_fill_tick_ack',      # W1 buy order fill price
       'ask_fill_tick_ack',      # W1 sell order fill price
       'best_bid_tick_ack',      # Best bid at order acknowledgment
       'best_ask_tick_ack',      # Best ask at order acknowledgment
       'bid_fill_tick_after_ack', # W2 buy order fill price
       'ask_fill_tick_after_ack'  # W2 sell order fill price
   }
   ```

---

## Backtesting Execution Flow

[accelerated_backtest](main.py#L9-L203) function's main loop logic:

### 1. Strategy Decision (Local Logic)
At each `local_ts[t]` timestamp:
```python
# Calculate mid price and position
mid_tick = (best_bid_tick[t] + best_ask_tick[t]) / 2.0
normalized_position = (position * mid_px) / max_notional_position

# Generate new orders based on strategy
req_bid_tick = ...  # Calculate buy order price
req_ask_tick = ...  # Calculate sell order price
```

### 2. Exchange Logic (Fill Determination)
Two cases:

#### Case A: Order Changes (Send new/modified orders)
```python
if req_bid_tick != open_bid_tick or req_ask_tick != open_ask_tick:
    # 1. Check if legacy orders are filled in W1 window
    if open_bid_tick >= bid_fill_tick_ack[t]:
        # Execute fill

    # 2. GTX check: reject orders that cross spread
    if req_bid_tick >= best_ask_tick_ack[t]:
        req_bid_tick = INVALID_MIN

    # 3. Accept new orders
    open_bid_tick = req_bid_tick

    # 4. Check if new orders are filled in W2 window
    if open_bid_tick >= bid_fill_tick_after_ack[t]:
        # Execute fill

    # 5. Move to next local_ts >= order_ack_ts[t]
    while local_ts[t] < order_ack_ts[t]:
        t += 1
```

#### Case B: No Order Changes (Maintain existing orders)
```python
else:
    # Check if legacy orders are filled in W0 window
    t += 1
    if open_bid_tick >= bid_fill_tick[t]:
        # Execute fill
```

### 3. State Record
Record current state each iteration:
```python
record[rec_i] = {
    'timestamp': local_ts[t],
    'price': mid_tick * tick_size,
    'position': position,
    'balance': balance * tick_size,
    'fee': trading_value * tick_size * fee,
    ...
}
```

---

## Usage Example

### Quick Start

```python
from datetime import UTC, datetime
from nanobt.utils import preprocess_data
from nanobt.stats import LinearAssetRecord

# 1. Preprocess data
table = preprocess_data(
    tick_size=0.1,           # Minimum price increment: 0.1 USDT
    lot_size=0.001,          # Minimum quantity increment: 0.001 BTC
    interval_ns=1_000_000_000,  # Strategy execution interval: 1 second
    start_date=datetime(2025, 11, 5, tzinfo=UTC),
    end_date=datetime(2025, 11, 10, tzinfo=UTC),
    base_dir="/share/tardis/raw/",  # Data directory
    exchange="binance-futures",
    symbol="BTCUSDT"
)

# 2. Run backtest
record = accelerated_backtest(
    relative_half_spread=0.00025,  # Half spread
    skew=0.00025,                  # Skew coefficient
    order_notional_value=50000,    # Order value
    max_notional_position=1000000, # Max position
    fee=-0.00005,                  # Fee rate (negative for rebates)
    tick_size=0.1,
    lot_size=0.001,
    **table.to_dict(as_series=False)  # Pass preprocessed data
)

# 3. Statistical analysis
stats = (
    LinearAssetRecord(record)
        .resample('1s')
        .stats(book_size=1000000)
)

print(stats.summary())  # Print statistics
fig = stats.plot()      # Plot equity curve
fig.savefig('backtest_result.png')
```

### Custom Strategy

Modify the strategy logic in [main.py:64-84](main.py#L64-L84):

```python
# Example: Fixed spread market making strategy
def custom_strategy(mid_tick, position, best_bid_tick, best_ask_tick):
    spread = 5  # Fixed 5-tick spread
    req_bid_tick = mid_tick - spread
    req_ask_tick = mid_tick + spread

    # Ensure orders don't cross spread
    req_bid_tick = min(req_bid_tick, best_bid_tick)
    req_ask_tick = max(req_ask_tick, best_ask_tick)

    return req_bid_tick, req_ask_tick
```

---

## Performance Optimization

### Numba JIT Compilation
All core functions use `@njit` decorator for machine code compilation:
```python
@njit
def accelerated_backtest(...):
    # First run compiles; subsequent calls run at near-C performance
```

### Memory Optimization
- Pre-allocate arrays with `np.empty` (see [main.py:59](main.py#L59))
- Convert prices to ticks (integers) to avoid floating-point errors
- Store records in Structured Arrays (see [types.py:39-51](src/nanobt/types.py#L39-L51))

### Parallelization Potential
- Different parameter combinations can be backtested in parallel (grid search)
- Polars data loading is automatically multi-threaded

---

## Limitations and Caveats

1. **No Queue Position Modeling**
   - May underestimate passive fill probability in low-liquidity or high-frequency strategies
   - Recommend comparing with full-order-book backtesting to validate critical strategies

2. **No Partial Fills**
   - Orders are either fully filled or not filled
   - Suitable for strategies where order size is much smaller than order book depth

3. **Simplified Latency Model**
   - Currently uses fixed latency; real network latency varies
   - Can be extended to historical-data-driven interpolation models

4. **Simplified Fee Model**
   - Example uses fixed fee rate (Maker rebate 0.005%)
   - Actual rate needs adjustment based on exchange VIP tier

---

## Dependencies

```bash
uv sync
```

Core dependencies:
- **NumPy**: Array operations
- **Numba**: JIT compilation acceleration
- **Polars**: High-performance data processing (5-10x faster than Pandas)

---

## File Structure

```
nanobt/
├── main.py                    # Backtest main loop + example strategy
├── src/nanobt/
│   ├── types.py              # Data type definitions (Schema, Constants)
│   ├── utils.py              # Data loading and preprocessing
│   └── stats/                # Statistical analysis module
│       ├── stats.py          # Stats class (metric calculation)
│       ├── metrics.py        # Various metric definitions
│       └── utils.py          # Helper functions
└── README.md
```