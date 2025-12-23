# NanoBT - 纳秒级加速回测框架

## 核心设计理念

### 加速原理：预计算成交条件

传统回测需要逐笔重放市场深度数据并模拟订单簿队列位置，而 NanoBT 采用**预计算填充价格**（Fill Price）的方法：

- **买单成交条件**：订单价格 ≥ 区间内最低卖价 或 > 最低主动卖成交价
- **卖单成交条件**：订单价格 ≤ 区间内最高买价 或 < 最高主动买成交价

**权衡**：不建模队列位置，订单要么完全成交要么不成交（无部分成交）。在高流动性市场（如主流加密货币合约）中，这种简化对大多数策略的精度影响可接受。

---

## 撮合逻辑详解

### 时间线切分：三个关键窗口

回测系统将时间线切分为三个窗口，分别处理不同状态的订单成交判定：

```
时间线：
    local_ts[t-1]      local_ts[t]         order_ack_ts[t]      local_ts[t+n]
        |                  |                     |                   |
        |<------ W0 ------>|<------- W1 -------->|<------ W2 ------->|
       遗留订单成交判定     撤单/改单来不及判定      新订单成交判定
```

#### **W0: 遗留订单窗口** `[local_ts[t-1], local_ts[t]]`
判断上一次策略运行后挂在交易所的订单（open orders）是否在当前时刻之前被成交。

- **输出数据**：
  - `bid_fill_tick`：买单成交价（tick），任何价格 ≥ 此值的挂单视为成交
  - `ask_fill_tick`：卖单成交价（tick），任何价格 ≤ 此值的挂单视为成交

- **计算方法**（见 [utils.py:417-420](src/nanobt/utils.py#L417-L420)）：
  ```python
  # 区间内最低主动卖成交价 + 1 tick，与最低卖一价取较小值
  bid_fill_tick = min(low_sell_tick + 1, low_best_ask_tick)
  # 区间内最高主动买成交价 - 1 tick，与最高买一价取较大值
  ask_fill_tick = max(high_buy_tick - 1, high_best_bid_tick)
  ```

#### **W1: 订单修改延迟窗口** `[local_ts[t], order_ack_ts[t]]`
本地在 `t` 时刻发出撤单/改单请求后，请求到达交易所前，原有订单是否会**抢先成交**（撤单/改单来不及）。

- **输出数据**：
  - `bid_fill_tick_ack`：买单延迟成交价（tick）
  - `ask_fill_tick_ack`：卖单延迟成交价（tick）
  - `best_bid_tick_ack`：交易所收到订单时的最优买价（用于 GTX 订单拒绝判定）
  - `best_ask_tick_ack`：交易所收到订单时的最优卖价（用于 GTX 订单拒绝判定）

- **应用场景**（见 [main.py:111-128](main.py#L111-L128)）：
  当策略试图撤销或修改订单时，需要判断在订单到达交易所之前是否已被成交：
  ```python
  # 检查挂单是否在订单到达交易所前成交
  if open_bid_tick > INVALID_MIN and open_bid_tick >= bid_fill_tick_ack[t]:
      # 买单成交：扣除资金，增加持仓
      balance -= open_bid_tick * open_bid_qty
      position += open_bid_qty
  ```

#### **W2: 新订单窗口** `[order_ack_ts[t], local_ts[t+n]]`
交易所确认订单后，到下一次策略观测点之间，新订单是否会被成交。

- **输出数据**：
  - `bid_fill_tick_after_ack`：新买单成交价（tick）
  - `ask_fill_tick_after_ack`：新卖单成交价（tick）

- **注意**：这里的 `local_ts[t+n]` 不一定是 `t+1`，因为订单延迟可能跨越多个本地时间步。

---

### 延迟模拟：订单确认时间戳

系统通过 `order_ack_ts` 模拟订单从本地发送到交易所确认的**单程延迟**（Entry Latency）：

```python
# utils.py:434
order_ack_ts = local_ts[t] + fixed_entry_lat_ns
```

- **默认延迟**：10ms（`fixed_entry_lat_ns = 10_000_000` 纳秒）
- **改进方向**：可扩展为基于历史数据的插值延迟模型（参考 HftBacktest 的 `IntpOrderLatency`）

**重要限制**：订单状态变化（受理/取消/成交/持仓更新）在**本地端立即生效**，不计入订单应答延迟。这是为了避免维护本地和交易所两套状态的复杂性。

---

### GTX 订单类型：防止立即成交

系统将所有限价单视为 **GTX（Good-Till-Crossing）** 订单，即"只挂单不吃单"：

```python
# main.py:130-138
# 如果买单价格 ≥ 卖一价，或卖单价格 ≤ 买一价，订单被拒绝
if req_bid_tick >= best_ask_tick_ack[t]:
    req_bid_tick = INVALID_MIN  # 拒绝订单
if req_ask_tick <= best_bid_tick_ack[t]:
    req_ask_tick = INVALID_MAX  # 拒绝订单
```

这避免了策略因"越过价差"而吃掉对手单，确保**只做 Maker 不做 Taker**。

---

### 订单状态管理：四个价格变量

系统使用四个变量追踪订单生命周期（见 [main.py:30-45](main.py#L30-L45)）：

```python
# 待确认的订单（本地已发送，交易所未确认）
req_bid_tick = INVALID_MIN  # 待确认买单价格（tick）
req_ask_tick = INVALID_MAX  # 待确认卖单价格（tick）

# 已确认的订单（交易所已接受，正在挂单）
open_bid_tick = INVALID_MIN  # 挂单中的买单价格（tick）
open_ask_tick = INVALID_MAX  # 挂单中的卖单价格（tick）
```

- **INVALID_MIN / INVALID_MAX**：特殊标记值，表示"无订单"或"订单已成交"
  - `INVALID_MIN = 0`：用于买单
  - `INVALID_MAX = 2^63 - 2`：用于卖单

**订单取消逻辑**（见 [main.py:38-40](main.py#L38-L40)）：
```python
# 如果 req_bid_tick 为 INVALID_MIN 且有挂单，则发送撤单请求
# （如果撤单请求在订单成交前到达交易所，订单会被取消）
if req_bid_tick == INVALID_MIN and open_bid_tick > INVALID_MIN:
    # 交易所将在 order_ack_ts[t] 处理撤单，前提是订单未在 W1 窗口成交
```

---

## 策略示例：偏度做市策略

内置的示例策略（[main.py:64-84](main.py#L64-L84)）实现了一个**仓位倾斜型双边做市策略**：

### 核心思路
1. 在中间价上下对称挂单做市
2. 根据当前持仓动态调整买卖挂单深度（偏度，Skew）
3. 达到风险限额时单边停止挂单

### 参数配置
```python
relative_half_spread = 0.00025  # 半价差：0.025%（挂单距中间价的基础距离）
skew = 0.00025                  # 偏度系数：根据持仓调整挂单距离
order_notional_value = 50000    # 单次订单名义价值：$50,000
max_notional_position = 1000000 # 最大持仓名义价值：$1,000,000（20倍杠杆）
```

### 挂单逻辑
```python
mid_tick = (best_bid_tick[t] + best_ask_tick[t]) / 2.0  # 中间价
normalized_position = (position * mid_px) / max_notional_position  # 归一化持仓 ∈ [-1, 1]

# 根据持仓偏度调整买卖深度
relative_bid_depth = relative_half_spread + skew * normalized_position
relative_ask_depth = relative_half_spread - skew * normalized_position

# 计算挂单价格（不能穿过盘口）
req_bid_tick = min(floor(mid_tick * (1 - relative_bid_depth)), best_bid_tick[t])
req_ask_tick = max(ceil(mid_tick * (1 + relative_ask_depth)), best_ask_tick[t])

# 风险控制：超过限额时停止对应方向挂单
if normalized_position > 1:  # 多仓超限
    req_bid_tick = INVALID_MIN  # 停止买入
if normalized_position < -1:  # 空仓超限
    req_ask_tick = INVALID_MAX  # 停止卖出
```

**偏度效果**：
- 持多仓时，买单深度增加（不激进买入），卖单深度减小（激进卖出平仓）
- 持空仓时，卖单深度增加（不激进卖出），买单深度减小（激进买入平仓）

---

## 数据预处理流程

### 输入数据要求

系统需要两类原始市场数据（Tardis 格式）：

1. **逐笔成交数据** (`trades`)
   ```python
   # 必需字段：timestamp, local_timestamp, side, price, amount
   side: 1 (buy) / -1 (sell)  # 主动买/卖方向
   ```

2. **最优报价数据** (`book_ticker`)
   ```python
   # 必需字段：timestamp, local_timestamp, bid_price, bid_amount, ask_price, ask_amount
   ```

### 预处理步骤

[preprocess_data](src/nanobt/utils.py#L552-L654) 函数的核心流程：

1. **加载数据**（支持 Parquet/CSV 格式）
   ```python
   trades = load_data(files, kind="trades")
   book_ticker = load_data(files, kind="book_ticker")
   ```

2. **生成本地时间戳序列**
   ```python
   # 以固定间隔（如 1 秒）生成策略运行时间点
   local_ts = np.arange(start_ts, end_ts, interval_ns)
   ```

3. **事件驱动迭代**（见 [utils.py:363-477](src/nanobt/utils.py#L363-L477)）

   使用 **Clock** 对象管理四个事件流：
   - `book_ticker_exch_clock`：交易所时间戳的报价更新
   - `book_ticker_local_clock`：本地时间戳的报价更新
   - `trades_exch_clock`：逐笔成交数据
   - `local_clock`：策略运行时间点

   **每次迭代**选择最早时间戳的事件处理：
   ```python
   ev = select_event([
       book_ticker_exch_clock.ts,
       book_ticker_local_clock.ts,
       trades_exch_clock.ts,
       local_clock.ts
   ])
   ```

4. **计算填充价格**

   每到达一个 `local_ts[t]`，调用 `ack_order` 函数计算三个窗口的填充价格：
   ```python
   # W0: 遗留订单
   bid_fill_tick = min(low_sell_tick + 1, low_best_ask_tick)
   ask_fill_tick = max(high_buy_tick - 1, high_best_bid_tick)

   # W1 & W2: 通过 ack_order 函数计算
   (bid_fill_tick_ack, ask_fill_tick_ack,
    best_bid_tick_ack, best_ask_tick_ack,
    bid_fill_tick_after_ack, ask_fill_tick_after_ack) = ack_order(...)
   ```

5. **输出预处理表**

   返回 Polars DataFrame，每行对应一个策略时间点：
   ```python
   {
       'local_ts',               # 本地时间戳
       'best_bid_tick',          # 本地最优买价
       'best_ask_tick',          # 本地最优卖价
       'bid_fill_tick',          # W0 买单成交价
       'ask_fill_tick',          # W0 卖单成交价
       'order_ack_ts',           # 订单确认时间戳
       'bid_fill_tick_ack',      # W1 买单成交价
       'ask_fill_tick_ack',      # W1 卖单成交价
       'best_bid_tick_ack',      # 订单确认时最优买价
       'best_ask_tick_ack',      # 订单确认时最优卖价
       'bid_fill_tick_after_ack', # W2 买单成交价
       'ask_fill_tick_after_ack'  # W2 卖单成交价
   }
   ```

---

## 回测执行流程

[accelerated_backtest](main.py#L9-L203) 函数的主循环逻辑：

### 1. 策略决策（本地逻辑）
在每个 `local_ts[t]` 时刻：
```python
# 计算中间价和持仓
mid_tick = (best_bid_tick[t] + best_ask_tick[t]) / 2.0
normalized_position = (position * mid_px) / max_notional_position

# 根据策略生成新订单
req_bid_tick = ...  # 计算买单价格
req_ask_tick = ...  # 计算卖单价格
```

### 2. 交易所逻辑（撮合判定）
分两种情况：

#### 情况 A：订单有变化（发出修改/新订单）
```python
if req_bid_tick != open_bid_tick or req_ask_tick != open_ask_tick:
    # 1. 检查遗留订单是否在 W1 窗口成交
    if open_bid_tick >= bid_fill_tick_ack[t]:
        # 执行成交

    # 2. GTX 检查：拒绝穿价订单
    if req_bid_tick >= best_ask_tick_ack[t]:
        req_bid_tick = INVALID_MIN

    # 3. 接受新订单
    open_bid_tick = req_bid_tick

    # 4. 检查新订单是否在 W2 窗口成交
    if open_bid_tick >= bid_fill_tick_after_ack[t]:
        # 执行成交

    # 5. 跳到下一个 local_ts >= order_ack_ts[t]
    while local_ts[t] < order_ack_ts[t]:
        t += 1
```

#### 情况 B：订单无变化（维持原有挂单）
```python
else:
    # 检查遗留订单是否在 W0 窗口成交
    t += 1
    if open_bid_tick >= bid_fill_tick[t]:
        # 执行成交
```

### 3. 状态记录
每次迭代记录当前状态：
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

## 使用示例

### 快速开始

```python
from datetime import UTC, datetime
from nanobt.utils import preprocess_data
from nanobt.stats import LinearAssetRecord

# 1. 预处理数据
table = preprocess_data(
    tick_size=0.1,           # 最小价格变动：0.1 USDT
    lot_size=0.001,          # 最小数量变动：0.001 BTC
    interval_ns=1_000_000_000,  # 策略运行间隔：1秒
    start_date=datetime(2025, 11, 5, tzinfo=UTC),
    end_date=datetime(2025, 11, 10, tzinfo=UTC),
    base_dir="/share/tardis/raw/",  # 数据目录
    exchange="binance-futures",
    symbol="BTCUSDT"
)

# 2. 运行回测
record = accelerated_backtest(
    relative_half_spread=0.00025,  # 半价差
    skew=0.00025,                  # 偏度系数
    order_notional_value=50000,    # 单次订单价值
    max_notional_position=1000000, # 最大持仓
    fee=-0.00005,                  # 手续费率（负数为返佣）
    tick_size=0.1,
    lot_size=0.001,
    **table.to_dict(as_series=False)  # 传入预处理数据
)

# 3. 统计分析
stats = (
    LinearAssetRecord(record)
        .resample('1s')
        .stats(book_size=1000000)
)

print(stats.summary())  # 打印统计指标
fig = stats.plot()      # 绘制权益曲线
fig.savefig('backtest_result.png')
```

### 自定义策略

修改 [main.py:64-84](main.py#L64-L84) 的策略逻辑：

```python
# 示例：固定价差做市策略
def custom_strategy(mid_tick, position, best_bid_tick, best_ask_tick):
    spread = 5  # 固定 5 个 tick 的价差
    req_bid_tick = mid_tick - spread
    req_ask_tick = mid_tick + spread

    # 确保不穿过盘口
    req_bid_tick = min(req_bid_tick, best_bid_tick)
    req_ask_tick = max(req_ask_tick, best_ask_tick)

    return req_bid_tick, req_ask_tick
```

---

## 性能优化

### Numba JIT 编译
所有核心函数使用 `@njit` 装饰器，编译为机器码：
```python
@njit
def accelerated_backtest(...):
    # 首次运行会编译，后续调用接近 C 语言性能
```

### 内存优化
- 使用 `np.empty` 预分配数组（见 [main.py:59](main.py#L59)）
- 价格统一转换为 tick（整数）避免浮点数误差
- Structured Array 存储记录数据（见 [types.py:39-51](src/nanobt/types.py#L39-L51)）

### 并行化潜力
- 不同参数组合可并行回测（网格搜索）
- Polars 数据加载自动多线程优化

---

## 局限性与注意事项

1. **无队列位置建模**
   - 在低流动性或高频策略中可能低估被动成交概率
   - 建议与完整回测对比验证关键策略

2. **无部分成交**
   - 订单要么完全成交要么不成交
   - 适用于订单量远小于盘口深度的策略

3. **延迟模型简化**
   - 当前使用固定延迟，真实网络延迟有波动
   - 可扩展为历史数据驱动的插值模型

4. **费用模型**
   - 示例使用固定费率（Maker 返佣 0.005%）
   - 实际需根据交易所 VIP 等级调整

---

## 依赖环境

```bash
uv sync
```

核心依赖：
- **NumPy**：数组运算
- **Numba**：JIT 编译加速
- **Polars**：高性能数据处理（比 Pandas 快 5-10 倍）

---

## 文件结构

```
nanobt/
├── main.py                    # 回测主流程 + 示例策略
├── src/nanobt/
│   ├── types.py              # 数据类型定义（Schema, Constants）
│   ├── utils.py              # 数据加载与预处理
│   └── stats/                # 统计分析模块
│       ├── stats.py          # Stats 类（指标计算）
│       ├── metrics.py        # 各类指标定义
│       └── utils.py          # 辅助函数
└── README.md
```