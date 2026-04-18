"""
Oil Price Shock → Market Reaction  |  Event Study + CAR Model
15.C51 Group Project

Framework
---------
1. Identify oil price shocks as new 3-year rolling highs/lows (deduplicated).
2. For each shock event t, estimate a baseline return model using an
   estimation window [t-270, t-30] (≈ 1 year, 30-day buffer before event):
     - S&P 500 : constant-mean model  (AR = R_sp - μ_sp)
     - XLE     : market model         (R_xle = α + β·R_sp + ε)
3. Compute abnormal returns (AR) and cumulative abnormal returns (CAR)
   over the event window [t-5, t+22].
4. Cross-sectional model: regress CAR(0→h) on shock-day features.
   Train on pre-2026 events, predict 2026 events, compare to actuals.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from scipy import stats

sns.set_theme(style="darkgrid", context="talk")
plt.rcParams["figure.dpi"] = 120

# ── Parameters ───────────────────────────────────────────────────────────────

ROLL_WINDOW      = 756   # 3-year rolling max/min for shock identification
MIN_GAP_DAYS     = 10    # minimum trading days between distinct events
EST_START        = -270  # estimation window start (days before event)
EST_END          = -30   # estimation window end   (days before event)
EVENT_PRE        = 5     # days before event to show
EVENT_POST       = 22    # days after event (longest CAR horizon)
CAR_HORIZONS     = [1, 5, 22]   # horizons for cross-sectional model

OUTDIR = "plots_event_study"
os.makedirs(OUTDIR, exist_ok=True)

def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, name), bbox_inches="tight")
    plt.close()
    print(f"  saved {OUTDIR}/{name}")

# ── 1. Load Brent crude ───────────────────────────────────────────────────────

oil = pd.read_csv("oil_data.csv")
oil.columns = [c.strip().lower().replace(".", "").replace(" ", "_").replace("%", "pct") for c in oil.columns]
oil.columns = [c.lstrip("\ufeff") for c in oil.columns]
oil = oil.rename(columns={"vol": "volume", "change_pct": "raw_chg_pct"})
for col in ["price", "open", "high", "low"]:
    oil[col] = pd.to_numeric(oil[col].astype(str).str.strip(), errors="coerce")
oil["date"]    = pd.to_datetime(oil["date"].astype(str).str.strip(), format="%m/%d/%Y")
oil            = oil.sort_values("date").reset_index(drop=True)
oil["ret_oil"] = oil["price"].pct_change()
oil            = oil.dropna(subset=["ret_oil"]).reset_index(drop=True)
oil["sigma_252"] = oil["ret_oil"].shift(1).rolling(252).std()

# ── 2. Download S&P 500 and XLE ───────────────────────────────────────────────

def download(ticker):
    raw = yf.download(ticker, start="1988-01-01", end="2026-04-17", progress=False)
    if raw.columns.nlevels > 1:
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    raw = raw.reset_index()
    raw.columns = [c.lower() for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"])
    col = ticker.lower().replace("^", "")
    raw[f"ret_{col}"] = raw["close"].pct_change()
    return raw[["date", "close", f"ret_{col}"]].rename(columns={"close": f"close_{col}"})

sp  = download("^GSPC")
xle = download("XLE")

# ── 3. Merge ──────────────────────────────────────────────────────────────────

df = (
    oil[["date", "price", "ret_oil", "sigma_252"]]
    .merge(sp,  on="date", how="inner")
    .merge(xle, on="date", how="inner")
)
df = df.sort_values("date").reset_index(drop=True)
print(f"Merged: {len(df):,} rows  {df['date'].min().date()} → {df['date'].max().date()}")

# ── 4. Shock identification (3-year rolling extreme + deduplication) ───────────

df["roll_max"]    = df["price"].rolling(ROLL_WINDOW).max()
df["roll_min"]    = df["price"].rolling(ROLL_WINDOW).min()
df["is_3y_high"]  = df["price"] >= df["roll_max"] * 0.999
df["is_3y_low"]   = df["price"] <= df["roll_min"] * 1.001
df["shock_raw"]   = df["is_3y_high"] | df["is_3y_low"]
df["shock_dir"]   = np.where(df["is_3y_high"], 1, np.where(df["is_3y_low"], -1, 0))
df["dist_from_max"] = (df["price"] - df["roll_max"]) / df["roll_max"]

candidates = df.index[df["shock_raw"]].tolist()

def extremeness(i):
    return df.loc[i, "price"] if df.loc[i, "is_3y_high"] else -df.loc[i, "price"]

candidates_sorted = sorted(candidates, key=lambda i: -extremeness(i))
accepted, suppressed = [], set()
for idx in candidates_sorted:
    if idx in suppressed:
        continue
    accepted.append(idx)
    for d in range(-MIN_GAP_DAYS, MIN_GAP_DAYS + 1):
        suppressed.add(idx + d)

df["shock"] = False
df.loc[accepted, "shock"] = True

shock_locs = [df.index.get_loc(i) for i in accepted]   # integer positions in df

print(f"\nDistinct events : {len(accepted)}")
print(f"  positive (3y high): {df.loc[accepted, 'shock_dir'].eq(1).sum()}")
print(f"  negative (3y low) : {df.loc[accepted, 'shock_dir'].eq(-1).sum()}")
print(f"  pre-2026: {df.loc[accepted, 'date'].dt.year.lt(2026).sum()}")
print(f"  2026    : {df.loc[accepted, 'date'].dt.year.ge(2026).sum()}")

# ── 5. Event study: estimate baseline models and compute CARs ─────────────────
#
# For each event at integer position `loc`:
#   Estimation window : rows [loc + EST_START, loc + EST_END)
#   Event window      : rows [loc - EVENT_PRE, loc + EVENT_POST]
#
# Baseline models:
#   S&P : constant-mean  →  E[R_sp] = mean(R_sp in estimation window)
#   XLE : market model   →  E[R_xle] = α̂ + β̂ · R_sp   (OLS in estimation window)

records = []

for loc in shock_locs:
    est_s = loc + EST_START
    est_e = loc + EST_END
    ev_s  = loc - EVENT_PRE
    ev_e  = loc + EVENT_POST

    # need enough data on both sides
    if est_s < 0 or ev_e >= len(df):
        continue

    est  = df.iloc[est_s:est_e]
    ev   = df.iloc[ev_s : ev_e + 1].copy()

    if est["ret_gspc"].isna().any() or est["ret_xle"].isna().any():
        continue
    if ev["ret_gspc"].isna().any() or ev["ret_xle"].isna().any():
        continue

    # S&P baseline: constant mean
    mu_sp  = est["ret_gspc"].mean()
    ar_sp  = ev["ret_gspc"].values - mu_sp

    # XLE baseline: market model  R_xle = α + β·R_sp
    X_est  = est["ret_gspc"].values.reshape(-1, 1)
    y_est  = est["ret_xle"].values
    mm     = LinearRegression().fit(X_est, y_est)
    ar_xle = ev["ret_xle"].values - mm.predict(ev["ret_gspc"].values.reshape(-1, 1))

    # CARs: cumsum over [EVENT_PRE, EVENT_PRE + h] (event day = index EVENT_PRE)
    # Index EVENT_PRE is t=0; indices 0..EVENT_PRE-1 are pre-event
    t0 = EVENT_PRE   # index of event day within event window arrays

    row = {
        "idx":       loc,
        "date":      df.iloc[loc]["date"],
        "price":     df.iloc[loc]["price"],
        "ret_oil":   df.iloc[loc]["ret_oil"],
        "shock_dir": df.iloc[loc]["shock_dir"],
        "sigma_252": df.iloc[loc]["sigma_252"],
        "dist_from_max": df.iloc[loc]["dist_from_max"],
        "beta_xle":  mm.coef_[0],
        # pre-shock S&P cumulative actual return (context feature)
        "sp_pre_22": df.iloc[loc - 22 : loc]["ret_gspc"].sum(),
        # full AR arrays for plotting
        "_ar_sp":    ar_sp,
        "_ar_xle":   ar_xle,
    }
    # CARs from day 0 onward
    for h in CAR_HORIZONS:
        row[f"car_sp_{h}"]  = ar_sp[t0 : t0 + h + 1].sum()
        row[f"car_xle_{h}"] = ar_xle[t0 : t0 + h + 1].sum()

    records.append(row)

events = pd.DataFrame(records)
print(f"\nEvents with sufficient data: {len(events)}")
print(f"  pre-2026 : {(events['date'].dt.year < 2026).sum()}")
print(f"  2026     : {(events['date'].dt.year >= 2026).sum()}")

# ── 6. Cross-sectional model: predict CAR from shock features ─────────────────
#
# Features (all known at shock day t, before event window outcome):
#   shock_dir     : +1 / -1
#   ret_oil       : magnitude and direction of oil return
#   sigma_252     : oil volatility regime
#   sp_pre_22     : S&P momentum prior 22d
#   dist_from_max : how far price is from the 3y max (supply slack signal)

FEATURES = ["shock_dir", "ret_oil", "sigma_252", "sp_pre_22", "dist_from_max"]

# Separate _ar arrays before splitting (they are used only for plots)
ar_arrays = events[["date", "_ar_sp", "_ar_xle", "shock_dir"]].copy()

train = events[events["date"].dt.year < 2026].dropna(subset=FEATURES + [f"car_sp_{h}" for h in CAR_HORIZONS]).copy()
test  = events[events["date"].dt.year >= 2026].dropna(subset=FEATURES).copy()

X_train = train[FEATURES].values
X_test  = test[FEATURES].values

scaler   = StandardScaler().fit(X_train)
Xs_train = scaler.transform(X_train)
Xs_test  = scaler.transform(X_test)

print(f"\nTraining events : {len(train)}")
print(f"Test events     : {len(test)}")

alphas = np.logspace(-3, 3, 60)
MODEL_SPECS = {
    "OLS":   lambda: LinearRegression(),
    "Ridge": lambda: RidgeCV(alphas=alphas, cv=5),
    "Lasso": lambda: LassoCV(alphas=alphas, cv=5, max_iter=10_000),
}
MODEL_COLORS = {"OLS": "forestgreen", "Ridge": "crimson", "Lasso": "purple"}

models  = {name: {} for name in MODEL_SPECS}
results = []

for name, make in MODEL_SPECS.items():
    for h in CAR_HORIZONS:
        target = f"car_sp_{h}"
        y_train = train[target].values

        m = make().fit(Xs_train, y_train)
        models[name][h] = m

        y_pred_train = m.predict(Xs_train)
        y_pred_test  = m.predict(Xs_test)

        cv_r2 = cross_val_score(m, Xs_train, y_train, cv=5, scoring="r2").mean()

        y_actual = test[target].values
        mask     = ~np.isnan(y_actual)
        test_mae = mean_absolute_error(y_actual[mask], y_pred_test[mask]) if mask.any() else np.nan

        coef = m.coef_ if hasattr(m, "coef_") else np.full(len(FEATURES), np.nan)
        results.append({
            "model":    name,
            "horizon":  h,
            "cv_r2":    cv_r2,
            "train_r2": r2_score(y_train, y_pred_train),
            "test_mae": test_mae,
            "coefs":    dict(zip(FEATURES, coef)),
        })
        test[f"car_sp_{h}_pred_{name}"] = y_pred_test

print("\n=== Cross-sectional model: CAR_SP ~ shock features ===")
print(f"{'Model':6s}  {'H':>3s}  {'cv_R²':>7s}  {'train_R²':>8s}  {'test_MAE':>10s}")
for r in results:
    mae_s = f"{r['test_mae']:.4f}" if pd.notna(r["test_mae"]) else "    n/a"
    print(f"{r['model']:6s}  {r['horizon']:3d}  {r['cv_r2']:7.3f}  {r['train_r2']:8.3f}  {mae_s:>10s}")

# ── 7. Plots ──────────────────────────────────────────────────────────────────

print("\nSaving plots...")

# ── 7a. Average CAR path (the canonical event-study plot) ─────────────────────

x = np.arange(-EVENT_PRE, EVENT_POST + 1)
t0 = EVENT_PRE

# Split by shock direction in training set
def extract_car_paths(subset, ar_col):
    paths = []
    for _, row in subset.iterrows():
        ar = row[ar_col]
        car = np.cumsum(ar) - np.cumsum(ar)[t0 - 1] if t0 > 0 else np.cumsum(ar)
        # normalise so that day -1 = 0  (CAR starts accumulating from t=0)
        car = np.cumsum(ar)
        car = car - car[t0 - 1] if t0 > 0 else car
        paths.append(car)
    return np.array(paths)

train_rows = events[events["date"].dt.year < 2026]
pos = train_rows[train_rows["shock_dir"] ==  1]
neg = train_rows[train_rows["shock_dir"] == -1]

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
combos = [
    (axes[0, 0], pos, "_ar_sp",  "S&P 500 – Positive shocks",  "darkorange"),
    (axes[0, 1], neg, "_ar_sp",  "S&P 500 – Negative shocks",  "steelblue"),
    (axes[1, 0], pos, "_ar_xle", "XLE – Positive shocks",      "darkorange"),
    (axes[1, 1], neg, "_ar_xle", "XLE – Negative shocks",      "steelblue"),
]
for ax, subset, ar_col, title, color in combos:
    if len(subset) == 0:
        continue
    paths = extract_car_paths(subset, ar_col)
    m, s  = paths.mean(0) * 100, paths.std(0) * 100
    se    = s / np.sqrt(len(paths))

    ax.plot(x, m, color=color, linewidth=2, label=f"Mean CAR  (n={len(paths)})")
    ax.fill_between(x, m - 2*se, m + 2*se, alpha=0.35, color=color, label="95% CI (mean)")
    ax.fill_between(x, m - s,   m + s,    alpha=0.12, color=color, label="±1 std (cross-section)")
    ax.axvline(0, linestyle="--", color="black", linewidth=1, label="Shock day")
    ax.axhline(0, linestyle=":",  color="black", linewidth=0.7)
    ax.set_title(title)
    ax.set_xlabel("Days relative to shock")
    ax.set_ylabel("Cumulative abnormal return (%)")
    ax.legend(fontsize=9)

plt.suptitle("Event Study: Cumulative Abnormal Returns around Oil Shocks\n"
             "(baseline: S&P constant-mean; XLE market model)", fontsize=13)
save("event_study_car.png")


# ── 7b. CAR distribution at each horizon (training set) ───────────────────────

fig, axes = plt.subplots(2, len(CAR_HORIZONS), figsize=(5 * len(CAR_HORIZONS), 9))
for col, h in enumerate(CAR_HORIZONS):
    for row, (asset, color) in enumerate([("sp", "steelblue"), ("xle", "darkorange")]):
        ax = axes[row][col]
        col_name = f"car_{asset}_{h}"
        data_pos = train_rows[train_rows["shock_dir"] ==  1][col_name].dropna() * 100
        data_neg = train_rows[train_rows["shock_dir"] == -1][col_name].dropna() * 100

        sns.kdeplot(data_pos, ax=ax, color="darkorange", fill=True, alpha=0.35, label=f"+ shock (n={len(data_pos)})")
        sns.kdeplot(data_neg, ax=ax, color="steelblue",  fill=True, alpha=0.35, label=f"− shock (n={len(data_neg)})")
        ax.axvline(data_pos.mean(), color="darkorange", linestyle="--", linewidth=1.2)
        ax.axvline(data_neg.mean(), color="steelblue",  linestyle="--", linewidth=1.2)
        ax.axvline(0, color="black", linewidth=0.7, linestyle=":")

        # t-test: is mean CAR different from zero?
        t_pos, p_pos = stats.ttest_1samp(data_pos, 0)
        t_neg, p_neg = stats.ttest_1samp(data_neg, 0)
        ax.set_title(f"{'S&P' if asset=='sp' else 'XLE'} CAR({h}d)\n"
                     f"+ mean={data_pos.mean():.2f}% p={p_pos:.2f}  "
                     f"− mean={data_neg.mean():.2f}% p={p_neg:.2f}")
        ax.set_xlabel("CAR (%)")
        ax.legend(fontsize=8)

plt.suptitle("Distribution of Cumulative Abnormal Returns by shock type (pre-2026)")
save("car_distributions.png")


# ── 7c. Cross-sectional scatter: ret_oil vs CAR (key relationship) ────────────

fig, axes = plt.subplots(1, len(CAR_HORIZONS), figsize=(5 * len(CAR_HORIZONS), 5))
for ax, h in zip(axes, CAR_HORIZONS):
    x_vals = train_rows["ret_oil"].values * 100
    y_vals = train_rows[f"car_sp_{h}"].values * 100
    colors = ["darkorange" if d == 1 else "steelblue" for d in train_rows["shock_dir"]]
    ax.scatter(x_vals, y_vals, c=colors, alpha=0.7, s=40, edgecolors="none")
    slope, intercept, r, p, _ = stats.linregress(x_vals, y_vals)
    xs = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="black", linewidth=1.5,
            label=f"OLS  β={slope:.2f}  R²={r**2:.3f}  p={p:.2f}")
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Oil return on shock day (%)")
    ax.set_ylabel(f"S&P CAR(0→{h}d) (%)")
    ax.set_title(f"Oil shock size vs S&P CAR ({h}d)")
    ax.legend(fontsize=9)
plt.suptitle("Cross-sectional relationship: oil shock magnitude → S&P abnormal return")
save("scatter_shock_vs_car.png")


# ── 7d. Coefficient heatmaps ─────────────────────────────────────────────────

for name in MODEL_SPECS:
    model_results = [r for r in results if r["model"] == name]
    coef_df = pd.DataFrame({f"{r['horizon']}d": r["coefs"] for r in model_results}).T
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(coef_df, annot=True, fmt=".4f", cmap="RdBu_r", center=0, ax=ax,
                linewidths=0.5, cbar_kws={"label": "coef (standardised features)"})
    ax.set_title(f"{name} – Coefficients for S&P CAR model")
    save(f"coefficients_{name.lower()}.png")


# ── 7e. Model comparison: CV R² across models and horizons ───────────────────

perf_df = pd.DataFrame(results)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric, label in [
    (axes[0], "cv_r2",    "5-fold CV R²"),
    (axes[1], "train_r2", "In-sample R²"),
]:
    pivot = perf_df.pivot(index="horizon", columns="model", values=metric)
    pivot.plot(kind="bar", ax=ax, rot=0, alpha=0.85,
               color=[MODEL_COLORS[c] for c in pivot.columns])
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Forecast horizon (days)")
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.legend(title="Model")
plt.suptitle("Model comparison – S&P CAR prediction")
save("model_comparison.png")


# ── 7f. 2026 predictions vs actuals  (THE MAIN RESULT) ───────────────────────

if len(test) > 0:
    n_models = len(MODEL_SPECS)
    n_events = len(test)
    fig, axes = plt.subplots(len(CAR_HORIZONS), 1, figsize=(max(12, n_events * 3), 5 * len(CAR_HORIZONS)))

    for ax, h in zip(axes, CAR_HORIZONS):
        target  = f"car_sp_{h}"
        actual  = test[target].values * 100
        dates   = test["date"].values
        dirs    = test["shock_dir"].values

        total_w = 0.75
        bar_w   = total_w / (n_models + 1)
        offsets = np.linspace(-total_w / 2, total_w / 2, n_models + 1)
        x_pos   = np.arange(n_events)
        has_act = ~np.isnan(actual)

        act_c = ["darkorange" if d == 1 else "steelblue" for d in dirs]
        if has_act.any():
            ax.bar(x_pos[has_act] + offsets[0], actual[has_act], bar_w,
                   color=[act_c[i] for i in np.where(has_act)[0]], alpha=0.85, label="Actual CAR")
        if (~has_act).any():
            ax.bar(x_pos[~has_act] + offsets[0], np.zeros((~has_act).sum()), bar_w,
                   color="grey", alpha=0.3, hatch="//", label="Actual (pending)")

        for i, name in enumerate(MODEL_SPECS):
            pred = test[f"{target}_pred_{name}"].values * 100
            ax.bar(x_pos + offsets[i + 1], pred, bar_w,
                   color=MODEL_COLORS[name], alpha=0.75, label=name)

        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [f"{pd.Timestamp(d).strftime('%b %d')} ({'↑' if di==1 else '↓'})"
             for d, di in zip(dates, dirs)],
            rotation=20, ha="right"
        )
        ax.set_ylabel("CAR (%)")
        ax.set_title(f"S&P CAR(0→{h}d) after 2026 oil shocks")
        ax.legend(fontsize=9)

        if has_act.any():
            parts = []
            for name in MODEL_SPECS:
                pred = test[f"{target}_pred_{name}"].values * 100
                mae  = mean_absolute_error(actual[has_act], pred[has_act])
                dacc = (np.sign(actual[has_act]) == np.sign(pred[has_act])).mean()
                parts.append(f"{name}: MAE={mae:.2f}pp dir={dacc:.0%}")
            ax.set_xlabel("  |  ".join(parts))

    plt.suptitle("2026 Oil Shock Events: Predicted vs Actual S&P Abnormal Returns", fontsize=13)
    save("predictions_2026.png")

    print("\n=== 2026 Events: Predicted vs Actual S&P CAR ===")
    out = test[["date", "price", "ret_oil", "shock_dir"]].copy()
    out["type"] = out["shock_dir"].map({1: "3y-high", -1: "3y-low"})
    for h in CAR_HORIZONS:
        out[f"actual_{h}d"] = (test[f"car_sp_{h}"] * 100).round(2)
        for name in MODEL_SPECS:
            out[f"{name}_{h}d"] = (test[f"car_sp_{h}_pred_{name}"] * 100).round(2)
    out = out.drop(columns=["shock_dir"])
    print(out.to_string(index=False))
else:
    print("\nNo 2026 test events found.")


# ── 7g. In-sample fit: actual CAR vs predicted CAR ───────────────────────────

dot_c = ["darkorange" if d == 1 else "steelblue" for d in train["shock_dir"]]
fig, axes = plt.subplots(len(MODEL_SPECS), len(CAR_HORIZONS),
                         figsize=(5 * len(CAR_HORIZONS), 4 * len(MODEL_SPECS)), squeeze=False)
for row, name in enumerate(MODEL_SPECS):
    for col, h in enumerate(CAR_HORIZONS):
        ax  = axes[row][col]
        y   = train[f"car_sp_{h}"].values * 100
        yh  = models[name][h].predict(Xs_train) * 100
        r   = next(r for r in results if r["model"] == name and r["horizon"] == h)
        ax.scatter(y, yh, c=dot_c, alpha=0.65, s=30, edgecolors="none")
        lo, hi = min(y.min(), yh.min()), max(y.max(), yh.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel("Actual CAR (%)")
        ax.set_ylabel("Predicted CAR (%)")
        ax.set_title(f"{name} – {h}d  train R²={r['train_r2']:.3f}  cv R²={r['cv_r2']:.3f}")
plt.suptitle("In-sample fit – pre-2026 training events\n(orange=positive shock, blue=negative)", y=1.01)
save("insample_fit.png")

print(f"\nAll plots saved to ./{OUTDIR}/")
