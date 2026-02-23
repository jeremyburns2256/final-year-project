"""
tf_arbitrage.py

Simple TensorFlow battery arbitrage model.
- Trains on JAN26 price data
- Tests on FEB26 price data
- Compares profit vs the threshold-based state machine baseline
"""

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Use your existing simulator
sys.path.insert(0, "utils")
from bess_simulator import simulate

# ── Config ─────────────────────────────────────────────────────────────────────
TRAIN_CSV   = "data/price_JAN26.csv"
TEST_CSV    = "data/price_FEB26.csv"
N_LAGS      = 12          # how many past prices to use as features (= 1 hour)
BUY_PCT     = 30          # label 'buy'  if price < this percentile of training set
SELL_PCT    = 70          # label 'sell' if price > this percentile of training set
EPOCHS      = 20
BATCH_SIZE  = 64
SEED        = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ── 1. Load data ───────────────────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

print(f"Train samples: {len(train_df)}  |  Test samples: {len(test_df)}")


# ── 2. Feature engineering ─────────────────────────────────────────────────────
def make_features(df: pd.DataFrame, n_lags: int) -> np.ndarray:
    """
    For each row t, build a feature vector of:
      - RRP at t-1, t-2, ..., t-n_lags   (lag prices)
      - rolling mean and std over those lags
    Returns array of shape (len(df) - n_lags, n_lags + 2).
    """
    rrp = df["RRP"].values.astype(float)
    rows = []
    for t in range(n_lags, len(rrp)):
        window = rrp[t - n_lags : t]
        rows.append([*window, window.mean(), window.std()])
    return np.array(rows, dtype=np.float32)


X_train = make_features(train_df, N_LAGS)
X_test  = make_features(test_df,  N_LAGS)

# Align dataframes to match features (drop first N_LAGS rows)
train_aligned = train_df.iloc[N_LAGS:].reset_index(drop=True)
test_aligned  = test_df.iloc[N_LAGS:].reset_index(drop=True)


# ── 3. Label generation ────────────────────────────────────────────────────────
# Labels are based on price percentiles of the *training* set only.
# This mimics "buy cheap, sell dear" without lookahead bias.
buy_thresh  = np.percentile(train_df["RRP"], BUY_PCT)
sell_thresh = np.percentile(train_df["RRP"], SELL_PCT)

print(f"\nLabel thresholds (from JAN26):")
print(f"  Buy  if RRP < {buy_thresh:.2f} $/MWh  ({BUY_PCT}th percentile)")
print(f"  Sell if RRP > {sell_thresh:.2f} $/MWh  ({SELL_PCT}th percentile)")
print(f"  Hold otherwise")

def label_actions(rrp_arr, buy_t, sell_t):
    """0 = buy, 1 = hold, 2 = sell"""
    labels = np.ones(len(rrp_arr), dtype=np.int32)   # default: hold
    labels[rrp_arr < buy_t]  = 0
    labels[rrp_arr > sell_t] = 2
    return labels

y_train = label_actions(train_aligned["RRP"].values, buy_thresh, sell_thresh)
y_test  = label_actions(test_aligned["RRP"].values,  buy_thresh, sell_thresh)

# Class distribution
unique, counts = np.unique(y_train, return_counts=True)
for cls, cnt in zip(["buy", "hold", "sell"], counts):
    print(f"  {cls}: {cnt} ({cnt/len(y_train)*100:.1f}%)")


# ── 4. Normalise features ──────────────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# ── 5. Build & train model ────────────────────────────────────────────────────
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax"),   # buy / hold / sell
], name="arbitrage_mlp")

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

print("\nTraining on JAN26...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1,
)

# Evaluate label accuracy on test set (how well JAN26 thresholds generalise)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFEB26 label accuracy: {test_acc*100:.1f}%  (vs JAN26-derived labels)")


# ── 6. Wrap predictions as a strategy for the simulator ───────────────────────
# Pre-compute all predictions for FEB26 so we don't call the model per-row.
probs   = model.predict(X_test, verbose=0)
actions = np.argmax(probs, axis=1)   # 0=buy, 1=hold, 2=sell
ACTION_MAP = {0: "buy", 1: "hold", 2: "sell"}

# Index into the pre-computed action array
_pred_idx = 0

def tf_strategy(rrp, battery_state):
    global _pred_idx
    action = ACTION_MAP[actions[_pred_idx]]
    _pred_idx = min(_pred_idx + 1, len(actions) - 1)
    return action


# ── 7. Simulate on FEB26 ──────────────────────────────────────────────────────
print("\nSimulating on FEB26 with TF model...")
results_tf = simulate(test_aligned, tf_strategy)

tf_profit = results_tf["cumulative_profit"].iloc[-1]
tf_cost   = results_tf["cumulative_cost"].iloc[-1]
tf_rev    = results_tf["cumulative_revenue"].iloc[-1]

print(f"\n{'─'*40}")
print(f"  TF Model Results (FEB26)")
print(f"{'─'*40}")
print(f"  Buy cost:    ${tf_cost:.2f}")
print(f"  Sell revenue:${tf_rev:.2f}")
print(f"  Net profit:  ${tf_profit:.2f}")

buy_c  = (results_tf["action"] == "buy").sum()
sell_c = (results_tf["action"] == "sell").sum()
hold_c = (results_tf["action"] == "hold").sum()
print(f"  Actions — buy:{buy_c}  sell:{sell_c}  hold:{hold_c}")


# ── 8. Baseline comparison (simple threshold on FEB26) ────────────────────────
def baseline_strategy(rrp, battery_state):
    if rrp < buy_thresh:
        return "buy"
    if rrp > sell_thresh:
        return "sell"
    return "hold"

print("\nSimulating on FEB26 with threshold baseline...")
results_base = simulate(test_aligned, baseline_strategy)

base_profit = results_base["cumulative_profit"].iloc[-1]
base_cost   = results_base["cumulative_cost"].iloc[-1]
base_rev    = results_base["cumulative_revenue"].iloc[-1]

print(f"\n{'─'*40}")
print(f"  Threshold Baseline Results (FEB26)")
print(f"{'─'*40}")
print(f"  Buy cost:    ${base_cost:.2f}")
print(f"  Sell revenue:${base_rev:.2f}")
print(f"  Net profit:  ${base_profit:.2f}")

buy_c  = (results_base["action"] == "buy").sum()
sell_c = (results_base["action"] == "sell").sum()
hold_c = (results_base["action"] == "hold").sum()
print(f"  Actions — buy:{buy_c}  sell:{sell_c}  hold:{hold_c}")

print(f"\n{'═'*40}")
print(f"  Improvement:  ${tf_profit - base_profit:+.2f}")
print(f"{'═'*40}")
