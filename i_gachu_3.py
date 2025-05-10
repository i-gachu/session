import os
import time
import json
import pandas as pd
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from pocketoptionapi.stable_api import PocketOption
import pocketoptionapi.global_value as global_value
from sklearn.ensemble import RandomForestClassifier


###RESIPOTORY 6 HOUR LIMIT, avoid ob and os, with trend###

# Load environment variables
load_dotenv()

# Session configuration
start_counter = time.perf_counter()

ssid = os.getenv("""SSID""")
demo = True

# Bot Settings
min_payout = 80
period = 300  
expiration = 300
INITIAL_AMOUNT = 1
MARTINGALE_LEVEL = 3
MIN_ACTIVE_PAIRS = 5
PROB_THRESHOLD = 0.76
TAKE_PROFIT = 20  # <-- Take profit target in dollars
current_profit = 0  # <-- Current cumulative profit

WATCHLIST = [
    "GBPAUD_otc", "GBPJPY_otc", "GBPUSD_otc",
    "AUDUSD_otc", "AUDCAD_otc", "CADCHF_otc",
    "USDCHF_otc", "USDJPY_otc", "USDCAD_otc",
]

# Connect to Pocket Option
api = PocketOption(ssid, demo)
api.connect()

FEATURE_COLS = ['RSI', 'k_percent', 'r_percent', 'MACD', 'MACD_EMA', 'Price_Rate_Of_Change']

# Utility Functions
def get_payout():
    try:
        d = json.loads(global_value.PayoutData)
        for pair in d:
            name = pair[1]
            payout = pair[5]
            if (
                name in WATCHLIST and
                pair[14] and
                name.endswith("_otc") and
                len(name) == 10
            ):
                if payout >= min_payout:
                    global_value.pairs[name] = {'payout': payout, 'type': pair[3]}
                elif name in global_value.pairs:
                    del global_value.pairs[name]
        return True
    except:
        return False

def get_df():
    try:
        for i, pair in enumerate(global_value.pairs, 1):
            df = api.get_candles(pair, period)
            global_value.logger(f'{pair} ({i}/{len(global_value.pairs)})', "INFO")
            time.sleep(1)
        return True
    except:
        return False

def make_df(df0, history):
    df1 = pd.DataFrame(history).sort_values(by='time').reset_index(drop=True)
    df1['time'] = pd.to_datetime(df1['time'], unit='s', utc=True)
    df1.set_index('time', inplace=True)
    df = df1['price'].resample(f'{period}s').ohlc().reset_index()
    if df0 is not None:
        ts = datetime.timestamp(df.loc[0]['time'])
        for x in range(len(df0)):
            ts2 = datetime.timestamp(df0.loc[x]['time'])
            if ts2 < ts:
                df = df._append(df0.loc[x], ignore_index=True)
            else:
                break
        df = df.sort_values(by='time').reset_index(drop=True)
    return df

def prepare_data(df):
    df = df[['time', 'open', 'high', 'low', 'close']]
    df.rename(columns={'time': 'timestamp'}, inplace=True)
    df.sort_values(by='timestamp', inplace=True)
    df['change_in_price'] = df['close'].diff()

    rsi_period = 14
    stochastic_period = 14
    macd_ema_long = 26
    macd_ema_short = 12
    macd_signal = 9
    roc_period = 9

    up_df = df['change_in_price'].where(df['change_in_price'] > 0, 0)
    down_df = abs(df['change_in_price'].where(df['change_in_price'] < 0, 0))
    ewma_up = up_df.ewm(span=rsi_period).mean()
    ewma_down = down_df.ewm(span=rsi_period).mean()
    rs = ewma_up / ewma_down
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

    df['low_14'] = df['low'].rolling(window=stochastic_period).min()
    df['high_14'] = df['high'].rolling(window=stochastic_period).max()
    df['k_percent'] = 100 * ((df['close'] - df['low_14']) / (df['high_14'] - df['low_14']))
    df['r_percent'] = ((df['high_14'] - df['close']) / (df['high_14'] - df['low_14'])) * -100

    ema_26 = df['close'].ewm(span=macd_ema_long).mean()
    ema_12 = df['close'].ewm(span=macd_ema_short).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_EMA'] = df['MACD'].ewm(span=macd_signal).mean()

    df['Price_Rate_Of_Change'] = df['close'].pct_change(periods=roc_period)
    df['Prediction'] = (df['close'].shift(-1) > df['close']).astype(int)

    df.dropna(inplace=True)
    return df

def train_and_predict(df):
    X_train = df[FEATURE_COLS].iloc[:-1]
    y_train = df['Prediction'].iloc[:-1]

    model = RandomForestClassifier(n_estimators=100, oob_score=True, criterion="gini", random_state=0)
    model.fit(X_train, y_train)

    X_test = df[FEATURE_COLS].iloc[[-1]]
    proba = model.predict_proba(X_test)
    call_conf = proba[0][1]
    put_conf = 1 - call_conf

    latest_close = df.iloc[-1]['close']
    latest_ema26 = df['close'].ewm(span=26).mean().iloc[-1]

    if call_conf > PROB_THRESHOLD and latest_close > latest_ema26:
        decision = "call"
        emoji = "🟢"
        confidence = call_conf
    elif put_conf > PROB_THRESHOLD and latest_close < latest_ema26:
        decision = "put"
        emoji = "🔴"
        confidence = put_conf
    else:
        global_value.logger("⏭️ Skipping trade due to low confidence or trend mismatch.", "INFO")
        return None

    global_value.logger(f"{emoji} === PREDICTED: {decision.upper()} | CONFIDENCE: {confidence:.2%}", "INFO")
    return decision

def perform_trade(amount, pair, action, expiration):
    result = api.buy(amount=amount, active=pair, action=action, expirations=expiration)
    trade_id = result[1]
    time.sleep(expiration)
    return api.check_win(trade_id)

def martingale_strategy(pair, action):
    global current_profit

    amount = INITIAL_AMOUNT
    level = 1
    result = perform_trade(amount, pair, action, expiration)

    if result is None:
        return

    if result[1] == 'win':
        current_profit += amount * (global_value.pairs[pair]['payout'] / 100)
        global_value.logger(f"✅ WIN - Profit: {current_profit:.2f} USD", "INFO")
    else:
        current_profit -= amount
        global_value.logger(f"❌ LOSS - Profit: {current_profit:.2f} USD", "INFO")

    while result[1] == 'loose' and level < MARTINGALE_LEVEL:
        level += 1
        amount *= 2
        result = perform_trade(amount, pair, action, expiration)

        if result is None:
            return

        if result[1] == 'win':
            current_profit += amount * (global_value.pairs[pair]['payout'] / 100)
            global_value.logger(f"✅ WIN - Profit: {current_profit:.2f} USD", "INFO")
            break
        else:
            current_profit -= amount
            global_value.logger(f"❌ LOSS - Profit: {current_profit:.2f} USD", "INFO")

    # ✅ Check Take Profit
    if current_profit >= TAKE_PROFIT:
        global_value.logger(f"🎯 Take Profit Achieved! Cooling down for 1 hour... Final Profit: {current_profit:.2f} USD", "INFO")
        time.sleep(3600)  # Sleep for 1 hour
        current_profit = 0  # Reset profit tracker after cooldown

    if result[1] != 'loose':
        global_value.logger("WIN - Resetting to base amount.", "INFO")
    else:
        global_value.logger("LOSS. Resetting.", "INFO")

def wait_until_next_candle(period_seconds=300, seconds_before=15):
    while True:
        now = datetime.now(timezone.utc)
        next_candle = ((now.timestamp() // period_seconds) + 1) * period_seconds
        if now.timestamp() >= next_candle - seconds_before:
            break
        time.sleep(0.2)

def wait_for_candle_start():
    while True:
        now = datetime.now(timezone.utc)
        if now.second == 0 and now.minute % (period // 60) == 0:
            break
        time.sleep(0.1)

# ✅ New timeout check function
def near_github_timeout():
    return (time.perf_counter() - start_counter) >= (6 * 3600 - 20 * 60)

# Strategy loop
def strategie():
    pairs_snapshot = list(global_value.pairs.keys())

    if len(pairs_snapshot) < MIN_ACTIVE_PAIRS:
        time.sleep(60)
        prepare()
        return

    for i, pair in enumerate(pairs_snapshot, 1):
        live_pairs = list(global_value.pairs.keys())
        if len(live_pairs) < MIN_ACTIVE_PAIRS:
            time.sleep(60)
            prepare()
            return

        if pair not in global_value.pairs:
            continue

        payout = global_value.pairs[pair].get('payout', 0)
        if payout < min_payout:
            continue

        wait_until_next_candle(period, 15)

        df = make_df(global_value.pairs[pair].get('dataframe'), global_value.pairs[pair].get('history'))
        if df is None or df.empty:
            continue

        global_value.logger(f"{len(df)} Candles collected for === {pair} === ({period // 60} mins timeframe)", "INFO")

        processed_df = prepare_data(df.copy())
        if processed_df.empty:
            continue

        decision = train_and_predict(processed_df)
       
        
        if decision:
            latest_rsi = processed_df.iloc[-1]['RSI']
            if (decision == "call" and latest_rsi > 70) or (decision == "put" and latest_rsi < 30):
                global_value.logger(f"Skipping {decision.upper()} due to RSI filter: RSI = {latest_rsi:.2f}", "INFO")
                continue

            if near_github_timeout():
                global_value.logger("🕒 Near GitHub timeout. Skipping new trade to avoid interruption.", "INFO")
                return
            wait_for_candle_start()
            martingale_strategy(pair, decision)

            get_payout()
            get_df()

def prepare():
    try:
        return get_payout() and get_df()
    except:
        return False

def start():
    while not global_value.websocket_is_connected:
        time.sleep(0.1)
    time.sleep(2)

    if prepare():
        while True:
            strategie()

# Main entry
if __name__ == "__main__":
    start()
    end_counter = time.perf_counter()
    global_value.logger(f"CPU-bound Task Time: {int(end_counter - start_counter)} seconds", "INFO")
