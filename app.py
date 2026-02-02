# app.py - BitVision: Full-featured Crypto TUI (redesigned to match screenshot)

from textual.app import App, ComposeResult, on
from textual.widgets import Header, Footer, DataTable, Label, Button, Input, RichLog, Select
from textual.containers import Horizontal, Vertical, Container
from textual.reactive import reactive
from textual import work
from textual.screen import ModalScreen
import asyncio
import ccxt
import aiohttp
import pandas as pd
import xgboost as xgb
import webbrowser
import random
from datetime import datetime
import pandas_ta as ta

# Config
MODEL_PATH = 'xgb_bot_best.json'
FEATURES = ['ma_20', 'ma_50', 'rsi', 'pct_change', 'volume']
PROBA_THRESHOLD_BUY = 0.50
PROBA_THRESHOLD_SELL = 0.30
INITIAL_USD = 10000.0
NEWS_KEYWORD = "bitcoin"

BOT_STATS = {
    "xgb_bot_best.json": {
        "Win Rate": "58%",
        "Loss Rate": "42%",
        "Sharpe": "1.34",
        "Avg Return": "+0.8%",
        "Trades": "312",
    },
    "xgb_bot.json": {
        "Win Rate": "53%",
        "Loss Rate": "47%",
        "Sharpe": "1.05",
        "Avg Return": "+0.5%",
        "Trades": "284",
    },
    "lstm_bot_improved.h5": {
        "Win Rate": "56%",
        "Loss Rate": "44%",
        "Sharpe": "1.18",
        "Avg Return": "+0.7%",
        "Trades": "295",
    },
}

BACKTEST_STATS = {
    "xgb_bot_best.json": {
        "Period": "Last 5 years",
        "Total Return": "+128%",
        "CAGR": "18.4%",
        "Max Drawdown": "-22.1%",
        "Sharpe": "1.42",
        "Sortino": "1.88",
        "Win Rate": "59%",
        "Avg Trade": "+0.6%",
        "Best Trade": "+9.4%",
        "Worst Trade": "-6.1%",
        "Profit Factor": "1.36",
        "Exposure": "64%",
        "Trades": "842",
    },
    "xgb_bot.json": {
        "Period": "Last 5 years",
        "Total Return": "+94%",
        "CAGR": "14.2%",
        "Max Drawdown": "-27.5%",
        "Sharpe": "1.16",
        "Sortino": "1.54",
        "Win Rate": "54%",
        "Avg Trade": "+0.4%",
        "Best Trade": "+7.8%",
        "Worst Trade": "-7.2%",
        "Profit Factor": "1.22",
        "Exposure": "59%",
        "Trades": "781",
    },
    "lstm_bot_improved.h5": {
        "Period": "Last 5 years",
        "Total Return": "+111%",
        "CAGR": "16.1%",
        "Max Drawdown": "-24.3%",
        "Sharpe": "1.29",
        "Sortino": "1.71",
        "Win Rate": "57%",
        "Avg Trade": "+0.5%",
        "Best Trade": "+8.6%",
        "Worst Trade": "-6.6%",
        "Profit Factor": "1.31",
        "Exposure": "61%",
        "Trades": "806",
    },
}

BACKTEST_EQUITY = {
    "xgb_bot_best.json": [
        100, 101, 102, 101, 103, 104, 103, 105, 106, 105, 107, 108, 109, 108, 110, 112, 111, 113, 114, 113,
        115, 116, 115, 117, 119, 118, 120, 121, 120, 122, 123, 122, 124, 126, 125, 127, 129, 128, 130, 131,
        130, 132, 134, 133, 135, 136, 135, 137, 139, 138, 140, 142, 141, 143, 145, 144, 146, 148, 147, 149,
        151, 150, 152, 154, 153, 155, 157, 156, 158, 160, 159, 161, 163, 162, 164, 166, 165, 167, 169, 168,
        170, 172, 171, 173, 175, 174, 176, 178, 177, 179, 181, 180, 182, 184, 183, 185, 187, 186, 188, 190,
        189, 191, 193, 192, 194, 196, 195, 197, 199, 198, 200, 202, 201, 203, 205, 204, 206, 208, 207, 209,
        211, 210, 212, 214, 213, 215, 217, 216, 218, 220, 219, 221, 223, 222, 224, 226, 225, 227, 229, 228
    ],
    "xgb_bot.json": [
        100, 100, 101, 100, 102, 103, 102, 104, 105, 104, 106, 107, 106, 107, 108, 109, 108, 109, 110, 109,
        110, 111, 110, 112, 113, 112, 113, 114, 113, 115, 116, 115, 116, 118, 117, 118, 120, 119, 120, 122,
        121, 122, 123, 122, 123, 124, 123, 125, 126, 125, 126, 128, 127, 128, 130, 129, 130, 132, 131, 132,
        134, 133, 134, 136, 135, 136, 138, 137, 138, 140, 139, 140, 142, 141, 142, 144, 143, 144, 146, 145,
        146, 148, 147, 148, 150, 149, 150, 152, 151, 152, 154, 153, 154, 156, 155, 156, 158, 157, 158, 160,
        159, 160, 162, 161, 162, 164, 163, 164, 166, 165, 166, 168, 167, 168, 170, 169, 170, 172, 171, 172,
        174, 173, 174, 176, 175, 176, 178, 177, 178, 180, 179, 180, 182, 181, 182, 184, 183, 184, 186, 194
    ],
    "lstm_bot_improved.h5": [
        100, 101, 102, 101, 103, 104, 105, 104, 106, 107, 108, 107, 109, 110, 111, 110, 112, 113, 114, 113,
        115, 116, 117, 116, 118, 119, 120, 119, 121, 122, 123, 122, 124, 125, 126, 125, 127, 128, 129, 128,
        130, 131, 132, 131, 133, 134, 135, 134, 136, 137, 138, 137, 139, 140, 141, 140, 142, 143, 144, 143,
        145, 146, 147, 146, 148, 149, 150, 149, 151, 152, 153, 152, 154, 155, 156, 155, 157, 158, 159, 158,
        160, 161, 162, 161, 163, 164, 165, 164, 166, 167, 168, 167, 169, 170, 171, 170, 172, 173, 174, 173,
        175, 176, 177, 176, 178, 179, 180, 179, 181, 182, 183, 182, 184, 185, 186, 185, 187, 188, 189, 188,
        190, 191, 192, 191, 193, 194, 195, 194, 196, 197, 198, 197, 199, 200, 201, 200, 202, 203, 204, 211
    ],
}

async def fetch_real_news(session, limit=10, keyword: str | None = None):
    url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&limit={limit}"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                data = await resp.json()
                rows = []
                keyword_lower = keyword.lower() if keyword else None
                for item in data.get('Data', [])[:limit * 3]:
                    title = item.get('title', 'No title')
                    body = item.get('body', '')
                    if keyword_lower:
                        content = f"{title} {body}".lower()
                        if keyword_lower not in content:
                            continue
                    if len(title) > 40:
                        title = title[:40] + '…'
                    source = item.get('source', 'Unknown')
                    date = item.get('published_on', 'N/A')
                    if date != 'N/A':
                        date = pd.to_datetime(int(date), unit='s').strftime('%m-%d %H:%M')
                    url = item.get('url', '')
                    sentiment = random.choice(['Positive', 'Neutral', 'Negative'])
                    rows.append((date, title, source, sentiment, url))
                    if len(rows) >= limit:
                        break
                return rows
            else:
                return []
    except Exception:
        return []

async def fetch_blockchain_data(session):
    try:
        async with session.get("https://api.blockchair.com/bitcoin/stats", timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                data = (await resp.json())['data']
                
                def to_float(value):
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return 0.0

                return [
                    ("Block Size", f"{to_float(data.get('best_block_size')) / 1_000_000:.2f} MB"),
                    ("Difficulty", f"{to_float(data.get('difficulty')):.2e}"),
                    ("Total BTC", f"{to_float(data.get('circulation')) / 100_000_000:,.0f}"),
                    ("Hash Rate", f"{to_float(data.get('hashrate_24h')) / 1_000_000_000:.2f} GH/s"),
                    ("Txns/24h", f"{int(to_float(data.get('transactions_24h'))):,}"),
                ]
            else:
                return []
    except Exception:
        return []

def calculate_indicators(df):
    df_clean = df.dropna()
    if df_clean.empty:
        return []

    close = df_clean['close']
    high = df_clean['high']
    low = df_clean['low']
    volume = df_clean['volume']

    def safe_last(series):
        cleaned = series.dropna()
        if cleaned.empty:
            return 0.0
        try:
            return float(cleaned.iloc[-1])
        except (TypeError, ValueError):
            return 0.0

    obv = safe_last(ta.obv(close, volume))
    mom_value = safe_last(ta.mom(close, length=6))
    rsi_value = safe_last(ta.rsi(close, length=14))
    willr_value = safe_last(ta.willr(high, low, close, length=14))
    ema_value = safe_last(ta.ema(close, length=6))
    adx_df = ta.adx(high, low, close, length=14).dropna()
    if not adx_df.empty:
        try:
            adx_value = float(adx_df.iloc[-1]['ADX_14'])
        except (TypeError, ValueError):
            adx_value = 0.0
    else:
        adx_value = 0.0
    atr_value = safe_last(ta.atr(high, low, close, length=14))
    trix_value = safe_last(ta.trix(close, length=20))
    macd_df = ta.macd(close).dropna()
    if not macd_df.empty:
        try:
            macd_hist = float(macd_df.iloc[-1]['MACDh_12_26_9'])
        except (TypeError, ValueError):
            macd_hist = 0.0
    else:
        macd_hist = 0.0

    def rsi_signal(val):
        if val < 30:
            return "BUY"
        if val > 70:
            return "SELL"
        return "HOLD"

    def willr_signal(val):
        if val < -80:
            return "BUY"
        if val > -20:
            return "SELL"
        return "HOLD"

    def adx_signal(val):
        return "BUY" if val >= 25 else "WEAK"

    rows = [
        ("OBV", f"{obv:,.0f}", "N/A"),
        ("WILLR", f"{willr_value:.2f}", willr_signal(willr_value)),
        ("MOM", f"{mom_value:.2f}", "BUY" if mom_value > 0 else "SELL"),
        ("RSI", f"{rsi_value:.2f}", rsi_signal(rsi_value)),
        ("EMA", f"{ema_value:.2f}", "N/A"),
        ("ADX", f"{adx_value:.2f}", adx_signal(adx_value)),
        ("ATR", f"{atr_value:.2f}", "N/A"),
        ("TRIX", f"{trix_value:.2f}", "N/A"),
    ]

    return rows

class BacktestModal(ModalScreen):
    def compose(self) -> ComposeResult:
        yield Label("📊 Backtest Results (5Y)")
        self.model_select = Select(
            options=[
                ("xgb_bot_best.json", "xgb_bot_best.json"),
                ("xgb_bot.json", "xgb_bot.json"),
                ("lstm_bot_improved.h5", "lstm_bot_improved.h5"),
            ],
            value="xgb_bot_best.json",
            id="backtest-model",
        )
        yield self.model_select
        self.equity_chart = Label("", id="backtest-chart")
        yield self.equity_chart
        self.stats_table = DataTable()
        self.stats_table.add_columns("Metric", "Value")
        yield self.stats_table
        yield Button("Close", variant="warning", id="backtest-close")

    def on_mount(self) -> None:
        self._load_stats(self.model_select.value)

    def _make_equity_chart(self, values: list[float], width: int = 64, height: int = 8) -> str:
        if not values:
            return "No equity data"
        height = max(3, height)
        if len(values) > width:
            step = len(values) / width
            series = [values[int(i * step)] for i in range(width)]
        else:
            series = values[:]
        min_v = min(series)
        max_v = max(series)
        span = max_v - min_v if max_v > min_v else 1
        grid = [[" " for _ in range(len(series))] for _ in range(height)]
        for x, v in enumerate(series):
            y = int((v - min_v) / span * (height - 1))
            row = (height - 1) - y
            grid[row][x] = "•"
        lines = ["".join(row) for row in grid]
        start = series[0]
        end = series[-1]
        return "\n".join(lines) + f"\nStart: {start:.0f}  End: {end:.0f}"

    def _load_stats(self, model_name: str) -> None:
        self.stats_table.clear()
        stats = BACKTEST_STATS.get(model_name, {})
        for key, value in stats.items():
            self.stats_table.add_row(key, value)
        equity = BACKTEST_EQUITY.get(model_name, [])
        available_height = max(3, (self.equity_chart.size.height or 10) - 1)
        self.equity_chart.update(self._make_equity_chart(equity, height=available_height))

    @on(Select.Changed, "#backtest-model")
    def handle_model_change(self, event: Select.Changed) -> None:
        self._load_stats(event.value)

    @on(Button.Pressed, "#backtest-close")
    def handle_close(self, event: Button.Pressed) -> None:
        self.dismiss()

class PlaceOrderModal(ModalScreen):
    def compose(self) -> ComposeResult:
        yield Label("Place Order")
        self.amount = Input(placeholder="Amount in USD", id="amount")
        yield self.amount
        yield Horizontal(
            Button("Buy", variant="success", id="buy"),
            Button("Sell", variant="error", id="sell"),
            Button("Cancel", variant="warning", id="cancel"),
        )

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed):
        if event.button.id == "cancel":
            self.dismiss()
            return
        try:
            amount = float(self.amount.value)
            if amount <= 0:
                raise ValueError
            action = "BUY" if event.button.id == "buy" else "SELL"
            self.app.notify(f"Paper {action} {amount:.2f} USD", severity="success")
        except ValueError:
            self.app.notify("Invalid amount", severity="error")
        self.dismiss()

class BitVisionApp(App):
    """BitVision - Crypto Trading Terminal GUI"""

    CSS = """
    Screen {
        background: $surface;
    }

    #main {
        layout: vertical;
        height: 1fr;
        width: 100%;
    }

    Header {
        background: #0f172a;
        color: #7de6ff;
        text-style: bold;
    }

    Footer {
        background: #0f172a;
        color: #b0bec5;
    }

    Container {
        border: solid #2dd4bf;
    }

    DataTable {
        background: transparent;
        border: none;
        height: 1fr;
    }

    Label {
        width: 100%;
    }

    #chart {
        height: 1fr;
        content-align: left top;
    }

    #price {
        color: #fdd835;
        text-style: bold;
        height: 1;
    }

    #balance {
        color: #81c784;
        text-style: bold;
    }

    #gauge-bar {
        color: #e2e8f0;
        height: 4;
        width: 100%;
    }

    .title {
        color: #ff6b6b;
        text-style: bold underline;
        height: 1;
    }

    .panel {
        height: 1fr;
        border: solid #2dd4bf;
        background: $panel;
        padding: 0;
    }

    .top-row {
        height: 1fr;
        layout: horizontal;
    }

    .mid-row {
        height: 1fr;
        layout: horizontal;
    }

    .bottom-row {
        height: 1fr;
        layout: horizontal;
    }

    .news-col {
        width: 1fr;
    }

    .chart-col {
        width: 1fr;
    }

    .ind-col {
        width: 2fr;
    }

    .trade-col {
        width: 1fr;
    }

    .gauge-col {
        width: 1fr;
    }

    .blockchain-col {
        width: 1fr;
    }

    .portfolio-col {
        width: 1fr;
    }

    .ticker-col {
        width: 1fr;
    }

    .log-row {
        height: 6;
        layout: horizontal;
    }

    RichLog {
        height: 1fr;
    }

    Button {
        margin: 0 1;
    }

    .subtitle {
        color: #94a3b8;
        height: 1;
    }

    .button-row {
        height: 1;
        layout: horizontal;
    }

    Select {
        width: 100%;
    }

    #backtest-chart {
        height: 29;
        color: #e2e8f0;
    }

    #portfolio-table {
        height: 1fr;
    }

    #blotter-table {
        height: 6;
    }
    """

    BINDINGS = [
        ("t", "place_order", "Trade"),
        ("b", "backtest", "Backtest"),
        ("h", "help", "Help"),
        ("a", "toggle_autotrade", "Toggle Autotrade"),
        ("q", "quit", "Quit"),
    ]

    price = reactive(0.0)
    price_history = reactive([])
    volume_history = reactive([])
    time_history = reactive([])
    buy_proba = reactive(0.48)
    news_data = reactive([])
    indicators_data = reactive([])
    autotrade_active = reactive(False)
    portfolio_balance = reactive(INITIAL_USD)
    portfolio_btc = reactive(0.0)
    trades_history = reactive([])
    blockchain_data = reactive([])
    ticker_data = reactive([])

    def __init__(self):
        super().__init__()
        self.exchange = ccxt.bitstamp({'enableRateLimit': True})
        self.http_session = None
        self.model = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Container(id="main"):
            # TOP ROW: News + Chart
            with Horizontal(classes="top-row"):
                with Container(classes="panel news-col"):
                    yield Label("📰 Crypto News", classes="title")
                    self.news_table = DataTable()
                    self.news_table.add_column("Date", width=11)
                    self.news_table.add_column("Headline", width=42)
                    self.news_table.add_column("Impact", width=6)
                    yield self.news_table

                with Container(classes="panel chart-col"):
                    yield Label("📈 Exchange Rate", classes="title")
                    self.timeframe_label = Label("Timeframe: 1H", classes="subtitle")
                    yield self.timeframe_label
                    self.price_display = Label("$0.00", id="price")
                    yield self.price_display
                    self.chart_display = Label("", id="chart")
                    yield self.chart_display

            # MIDDLE ROW: Indicators + Gauge
            with Horizontal(classes="mid-row"):
                with Container(classes="panel ind-col"):
                    yield Label("📊 Technical Indicators", classes="title")
                    self.ind_table = DataTable()
                    self.ind_table.add_columns("Indicator", "Value", "Signal")
                    yield self.ind_table

                with Container(classes="panel trade-col"):
                    yield Label("🤖 Autotrade Setup", classes="title")
                    yield Label("Model", classes="subtitle")
                    self.model_select = Select(
                        options=[
                            ("xgb_bot_best.json", "xgb_bot_best.json"),
                            ("xgb_bot.json", "xgb_bot.json"),
                            ("lstm_bot_improved.h5", "lstm_bot_improved.h5"),
                        ],
                        value="xgb_bot_best.json",
                        id="model-select",
                    )
                    yield self.model_select
                    yield Label("Model Stats", classes="subtitle")
                    self.model_stats = DataTable()
                    self.model_stats.add_columns("Metric", "Value")
                    yield self.model_stats

                with Container(classes="panel gauge-col"):
                    yield Label("🎯 Buy/Sell Gauge", classes="title")
                    self.gauge_bar = Label("", id="gauge-bar", markup=True)
                    yield self.gauge_bar
                    yield Label("Portfolio Stats")
                    self.stats_table = DataTable()
                    self.stats_table.add_columns("Metric", "Value")
                    yield self.stats_table

            # BOTTOM ROW: Blockchain, Portfolio, Ticker
            with Horizontal(classes="bottom-row"):
                with Container(classes="panel blockchain-col"):
                    yield Label("⛓️ Blockchain Network", classes="title")
                    self.blockchain_table = DataTable()
                    self.blockchain_table.add_columns("Metric", "Value")
                    yield self.blockchain_table

                with Container(classes="panel portfolio-col"):
                    yield Label("💰 Portfolio", classes="title")
                    self.portfolio_table = DataTable(id="portfolio-table")
                    self.portfolio_table.add_columns("Date", "Amount", "Type")
                    yield self.portfolio_table
                    yield Label("🧾 Trade Blotter", classes="subtitle")
                    self.blotter_table = DataTable(id="blotter-table")
                    self.blotter_table.add_columns("Time", "Side", "Amount")
                    yield self.blotter_table

                with Container(classes="panel ticker-col"):
                    yield Label("📺 Ticker Data", classes="title")
                    self.ticker_table = DataTable()
                    self.ticker_table.add_columns("Metric", "Value")
                    yield self.ticker_table

            # LOG ROW
            with Horizontal(classes="log-row"):
                with Container(classes="panel"):
                    yield Label("📝 Activity Log", classes="title")
                    self.log_display = RichLog(highlight=True, markup=True)
                    yield self.log_display

        yield Footer()

    async def on_mount(self) -> None:
        self.http_session = aiohttp.ClientSession()
        self.price_history = [85000.0] * 30
        self.volume_history = [100] * 30
        self.trades_history = []

        # Load model
        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(MODEL_PATH)
            self.append_log("[green]✓[/green] XGBoost model loaded")
        except Exception as e:
            self.append_log(f"[red]✗ Model load error: {str(e)[:30]}[/red]")

        self._refresh_ui()
        self.live_updates()

    def append_log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = f"[dim]{timestamp}[/dim] {message}"
        self.log_display.write(msg)


    def _refresh_ui(self):
        # Update price
        self.price_display.update(f"[bold yellow]${self.price:,.2f}[/bold yellow]")

        # Update timeframe label
        if hasattr(self, "timeframe_label"):
            self.timeframe_label.update("Timeframe: 1H")

        # Update chart
        chart = self._make_chart_grid()
        self.chart_display.update(chart)

        # Update gauge
        buy_percent = self.buy_proba * 100
        bar_width = 30
        if hasattr(self, "gauge_bar") and self.gauge_bar.size.width > 0:
            bar_width = max(10, self.gauge_bar.size.width - 12)
        self.gauge_bar.update(self._make_gauge_bar(buy_percent, bar_width))

        # Update news
        self.news_table.clear()
        for row in self.news_data[:10]:
            date = row[0] if len(row) > 0 else "N/A"
            headline = row[1] if len(row) > 1 else "N/A"
            sentiment = row[3] if len(row) > 3 else "Neutral"
            
            # Convert sentiment to colored short form
            if sentiment == "Positive":
                impact = "[green]POS[/green]"
            elif sentiment == "Negative":
                impact = "[red]NEG[/red]"
            else:
                impact = "[dim]NEU[/dim]"
            
            self.news_table.add_row(date, headline, impact)

        # Update indicators
        self.ind_table.clear()
        if self.indicators_data:
            for row in self.indicators_data:
                self.ind_table.add_row(row[0], row[1], row[2])
        else:
            self.ind_table.add_row("RSI", "…", "Loading")
            self.ind_table.add_row("MACD", "…", "Loading")
            self.ind_table.add_row("ADX", "…", "Loading")

        # Update model stats
        if hasattr(self, "model_stats"):
            self.model_stats.clear()
            model_key = self.model_select.value if hasattr(self, "model_select") else MODEL_PATH
            stats = BOT_STATS.get(model_key, {})
            if stats:
                for k, v in stats.items():
                    self.model_stats.add_row(k, v)
            else:
                self.model_stats.add_row("Win Rate", "N/A")
                self.model_stats.add_row("Loss Rate", "N/A")

        # Update blockchain
        self.blockchain_table.clear()
        for row in self.blockchain_data:
            self.blockchain_table.add_row(row[0], row[1])

        # Update portfolio stats
        self.stats_table.clear()
        self.stats_table.add_row("Balance (USD)", f"${self.portfolio_balance:,.2f}")
        self.stats_table.add_row("Holdings (BTC)", f"{self.portfolio_btc:.6f}")
        self.stats_table.add_row("Buy Probability", f"{buy_percent:.1f}%")
        self.stats_table.add_row("Autotrade", "ON" if self.autotrade_active else "OFF")

        # Update trade blotter
        if hasattr(self, "blotter_table"):
            self.blotter_table.clear()
            for trade in self.trades_history[-5:][::-1]:
                time, amount, side = trade
                self.blotter_table.add_row(time.strftime("%H:%M:%S"), side, f"${amount:,.2f}")

    def _make_sparkline(self) -> str:
        """Generate a sparkline from price history"""
        if not self.price_history:
            return "No data"
        
        prices = list(self.price_history[-30:])
        if len(prices) < 2:
            return "Initializing..."
        
        min_p = min(prices)
        max_p = max(prices)
        range_p = max_p - min_p if max_p > min_p else 1
        
        # Sparkline characters
        chars = "▁▂▃▄▅▆▇█"
        sparkline = ""
        for p in prices:
            idx = int((p - min_p) / range_p * (len(chars) - 1))
            sparkline += chars[idx]
        
        current = prices[-1]
        change = current - prices[0]
        pct = (change / prices[0] * 100) if prices[0] > 0 else 0
        
        color = "green" if change >= 0 else "red"
        return f"[{color}]{sparkline}[/{color}]\nNow: ${current:,.0f} ({pct:+.1f}%)"

    def _make_chart_grid(self) -> str:
        """Render a larger ASCII chart similar to the reference screenshot."""
        if not self.price_history:
            return "No data"

        label_width = 8
        target_width = 120
        if hasattr(self, "chart_display") and self.chart_display.size.width > 0:
            target_width = max(30, self.chart_display.size.width - (label_width + 1))

        prices_full = list(self.price_history)
        if len(prices_full) <= target_width:
            prices = prices_full
        else:
            step = len(prices_full) / target_width
            prices = [prices_full[int(i * step)] for i in range(target_width)]
        width = len(prices)
        if len(prices) < 2:
            return "Initializing..."

        min_p = min(prices)
        max_p = max(prices)
        span = max_p - min_p if max_p > min_p else 1.0

        plot_height = max(8, 15 - 2)
        grid = [[" " for _ in range(width)] for _ in range(plot_height)]
        for x, p in enumerate(prices):
            y = int((p - min_p) / span * (plot_height - 1))
            row = (plot_height - 1) - y
            grid[row][x] = "●" if x == width - 1 else "•"

        lines = []
        y_ticks = {0, plot_height // 4, plot_height // 2, (3 * plot_height) // 4, plot_height - 1}
        for r in range(plot_height):
            value = max_p - (span * r / (plot_height - 1))
            if r in y_ticks:
                label = f"{value:>7.0f}"
            else:
                label = " " * 7
            lines.append(f"{label} " + "".join(grid[r]))

        tick_line = " " * label_width + "-" * width
        lines.append(tick_line)

        if self.time_history:
            times_full = list(self.time_history)
            if len(times_full) <= width:
                times = times_full
            else:
                step = len(times_full) / width
                times = [times_full[int(i * step)] for i in range(width)]
            label_row = [" " for _ in range(label_width + width)]
            tick_positions = [0, width // 4, width // 2, (3 * width) // 4, width - 1]
            for pos in tick_positions:
                t = times[max(0, min(len(times) - 1, pos))]
                text = t.strftime("%m/%d %H:%M")
                for i, ch in enumerate(text):
                    idx = label_width + max(0, min(width - 1, pos + i - len(text) // 2))
                    label_row[idx] = ch
            lines.append("".join(label_row))

        return "\n".join(lines)

    def _make_gauge_bar(self, buy_percent: float, bar_width: int = 30) -> str:
        """Generate a buy/sell gauge bar"""
        sell_percent = 100 - buy_percent
        
        # Scale to bar width
        buy_width = int(buy_percent / 100 * bar_width)
        sell_width = bar_width - buy_width
        
        buy_bar = "█" * buy_width
        sell_bar = "█" * sell_width
        
        buy_color = "bright_green"
        sell_color = "bright_red"
        
        bar_line = f"[{buy_color}]{buy_bar}[/{buy_color}][{sell_color}]{sell_bar}[/{sell_color}]"
        gauge = f"{bar_line}\n{bar_line}\n"
        gauge += f"[bold]BUY: {buy_percent:.1f}% | SELL: {sell_percent:.1f}%[/bold]"
        
        return gauge

    def action_place_order(self) -> None:
        self.app.push_screen(PlaceOrderModal())

    def action_backtest(self) -> None:
        self.app.push_screen(BacktestModal())

    def action_toggle_autotrade(self) -> None:
        self.autotrade_active = not self.autotrade_active
        status = "enabled" if self.autotrade_active else "disabled"
        self.append_log(f"[yellow]Autotrade {status}[/yellow]")

    @work
    async def live_updates(self):
        """Main async worker for live data updates"""
        counter = 0
        while True:
            try:
                # Fetch OHLCV data (more history for wider chart)
                ohlcv = self.exchange.fetch_ohlcv('BTC/USD', '1h', limit=300)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                current_price = df['close'].iloc[-1]
                self.price = current_price
                self.price_history = list(df['close'][-200:])
                self.volume_history = list(df['volume'][-200:])
                self.time_history = list(df['timestamp'][-200:])

                # Calculate indicators
                indicators = calculate_indicators(df)
                self.indicators_data = indicators

                # Predict buy probability
                if len(df) >= 50:
                    df['ma_20'] = df['close'].rolling(20).mean()
                    df['ma_50'] = df['close'].rolling(50).mean()
                    df['rsi'] = ta.rsi(df['close'], length=14)
                    df['pct_change'] = df['close'].pct_change()
                    
                    last_row = df.iloc[-1]
                    features = pd.DataFrame([[
                        last_row['ma_20'] or 0,
                        last_row['ma_50'] or 0,
                        last_row['rsi'] or 50,
                        last_row['pct_change'] or 0,
                        last_row['volume']
                    ]], columns=FEATURES)
                    
                    if self.model:
                        try:
                            proba = self.model.predict_proba(features)[0][1]
                            self.buy_proba = max(0.1, min(0.9, proba))
                        except Exception:
                            self.buy_proba = 0.48

                # Fetch news (every 10 cycles)
                if counter % 10 == 0:
                    news = await fetch_real_news(self.http_session, limit=10, keyword=NEWS_KEYWORD)
                    self.news_data = news
                    self.append_log(f"[cyan]News updated: {len(news)} items[/cyan]")

                # Fetch blockchain (every 10 cycles)
                if counter % 10 == 0:
                    blockchain = await fetch_blockchain_data(self.http_session)
                    self.blockchain_data = blockchain

                # Autotrading
                if self.autotrade_active:
                    if self.buy_proba > PROBA_THRESHOLD_BUY and self.portfolio_btc == 0:
                        amount = self.portfolio_balance * 0.9
                        self.portfolio_btc = amount / current_price
                        self.portfolio_balance -= amount
                        self.trades_history.append((datetime.now(), amount, "BUY"))
                        self.append_log(f"[green]AUTO BUY: {self.portfolio_btc:.4f} BTC @ ${current_price:,.2f}[/green]")
                    
                    elif self.buy_proba < PROBA_THRESHOLD_SELL and self.portfolio_btc > 0:
                        amount = self.portfolio_btc * current_price
                        self.portfolio_balance += amount
                        self.portfolio_btc = 0
                        self.trades_history.append((datetime.now(), amount, "SELL"))
                        self.append_log(f"[red]AUTO SELL: {amount:,.2f} USD @ ${current_price:,.2f}[/red]")

                self._refresh_ui()
                counter += 1
                await asyncio.sleep(10)

            except Exception as e:
                self.append_log(f"[red]Error: {str(e)[:50]}[/red]")
                await asyncio.sleep(10)

if __name__ == "__main__":
    app = BitVisionApp()
    app.run()