import ccxt
import pandas as pd

exchange = ccxt.bitstamp({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_API_SECRET',
})

def fetch_historical_data(symbol='BTC/USD', timeframe='1h', limit=1000):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Usage: data = fetch_historical_data()