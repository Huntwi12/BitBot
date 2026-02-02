import ccxt
import time
import pandas as pd
import numpy as np
import xgboost as xgb

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
MODEL_PATH = 'xgb_bot_best.json'
PROBA_THRESHOLD = 0.48          # Tune this! Start 0.45–0.50
SYMBOL = 'BTC/USD'
TIMEFRAME = '1h'
CHECK_INTERVAL_SECONDS = 3600   # 1 hour
PAPER_MODE = True
INITIAL_PAPER_USD = 10000
TRADE_SIZE_PCT = 0.95           # Use 95% of available USD for safety

# Load model
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

FEATURES = ['ma_20', 'ma_50', 'rsi', 'pct_change', 'volume']

# ... rest of your trader.py ...

class Trader:
    def __init__(self, exchange, symbol='BTC/USD', paper_mode=True, initial_balance=10000):
        # ...
        self.proba_threshold = 0.48   # ← start here; adjust after paper testing
        self.model = xgb.XGBClassifier()
        self.model.load_model('xgb_bot_best.json')  # ← use XGBoost

    def predict_buy_proba(self):
        X = self.get_latest_features()
        return self.model.predict_proba(X)[0, 1]

    def run(self):
        while True:
            try:
                proba = self.predict_buy_proba()
                current_price = self.exchange.fetch_ticker(self.symbol)['last']
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

                print(f"[{timestamp}] BTC/USD: ${current_price:.2f} | Buy Proba: {proba:.4f}")

                if proba > self.proba_threshold:
                    if self.paper_mode:
                        usd_avail = self.balance['USD']
                        amount_btc = (usd_avail * 0.95) / current_price  # 5% margin
                        cost_usd = amount_btc * current_price
                        self.balance['BTC'] += amount_btc
                        self.balance['USD'] -= cost_usd
                        print(f"  [PAPER BUY] {amount_btc:.6f} BTC @ ${current_price:.2f} | "
                              f"New: USD {self.balance['USD']:.2f} | BTC {self.balance['BTC']:.6f}")
                    else:
                        # Real trade – start VERY small!
                        amount = 0.0005  # e.g. ~$30–50 at $60k BTC – CHANGE CAREFULLY
                        self.exchange.create_market_buy_order(self.symbol, amount)
                        print(f"  [LIVE BUY] {amount} BTC executed")

                time.sleep(3600)  # hourly check
            except Exception as e:
                print(f"Loop error: {str(e)}")
                time.sleep(60)
if __name__ == "__main__":
    exchange = ccxt.bitstamp({
        'apiKey': '',
        'secret': '',
    })
    trader = Trader(exchange, paper_mode=True)
    trader.run()