import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
import joblib
import xgboost as xgb
import time

def fetch_historical_data(symbol='BTC/USD', timeframe='1h', limit=2000):
    exchange = ccxt.bitstamp()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def add_features(df):
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['pct_change'] = df['close'].pct_change()
    df = df.dropna()
    return df

def label_data(df, lookahead=1, threshold=0.002):
    df['future_close'] = df['close'].shift(-lookahead)
    df['future_return'] = (df['future_close'] - df['close']) / df['close']
    df['target'] = (df['future_return'] > threshold).astype(int)
    return df.dropna()

def train_model(df, model_type='xgboost', threshold=0.002, smote_ratio=0.4):
    features = ['ma_20', 'ma_50', 'rsi', 'pct_change', 'volume']

    df = label_data(df, threshold=threshold)

    print(f"\nClass distribution (original):")
    print(df['target'].value_counts(normalize=True).round(4))

    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # SMOTE only on training set
    if smote_ratio > 0:
        smote = SMOTE(random_state=42, sampling_strategy=smote_ratio)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"Training class distribution after SMOTE: {np.bincount(y_train_res)}")
    else:
        X_train_res, y_train_res = X_train, y_train

    if model_type == 'randomforest':
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample'
        )
        model.fit(X_train_res, y_train_res)
        probas = model.predict_proba(X_test)[:, 1]
        name = "Random Forest"

    elif model_type == 'xgboost':
        scale_pos = sum(y_train == 0) / sum(y_train == 1) if sum(y_train == 1) > 0 else 1
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.03,
            scale_pos_weight=scale_pos,
            random_state=42,
            n_jobs=-1,
            eval_metric='aucpr'
        )
        model.fit(X_train_res, y_train_res)
        probas = model.predict_proba(X_test)[:, 1]
        name = "XGBoost"

    # Evaluate with multiple thresholds
    print(f"\n{name} Results (on original test set):")
    print(f"ROC-AUC: {roc_auc_score(y_test, probas):.4f}" if len(np.unique(y_test)) > 1 else "ROC-AUC: nan (single class in test)")

    print("\nThreshold tuning for buy signals:")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds = (probas > thresh).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', zero_division=0)
        support = sum(y_test == 1)
        print(f"  >{thresh}: Prec/Rec/F1/Support = {prec:.4f} / {rec:.4f} / {f1:.4f} / {support}")
        if thresh == 0.5:
            print("  Confusion Matrix (at 0.5):\n", confusion_matrix(y_test, preds))

    # Save best model (using default 0.5 threshold)
    if model_type == 'randomforest':
        joblib.dump(model, 'rf_bot_best.pkl')
        print("→ Saved: rf_bot_best.pkl")
    else:
        model.save_model('xgb_bot_best.json')
        print("→ Saved: xgb_bot_best.json")

def main():
    print("Script starting...")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    print("Fetching data...")
    df = fetch_historical_data(limit=2000)
    print(f"Raw shape: {df.shape}")

    print("Adding features...")
    df = add_features(df)
    print(f"After features: {df.shape}")

    print("\nTraining XGBoost (recommended)...")
    train_model(df, model_type='xgboost', threshold=0.002, smote_ratio=0.3)

    print("\nTraining Random Forest...")
    train_model(df, model_type='randomforest', threshold=0.002, smote_ratio=0.3)

    print("\nDone.")

if __name__ == "__main__":
    main()