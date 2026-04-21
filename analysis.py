"""
Oil Price Shock → Market Reaction  |  Event Study + Multi-Sector + Advanced ML
15.C51 Group Project

Framework
---------
1. Identify oil price shocks as new 3-year rolling highs/lows (deduplicated).
2. For each shock event t, estimate a baseline return model using an
   estimation window [t-270, t-30]:
     - S&P 500 : constant-mean model
     - All SPDR sectors: market model (R_sec = α + β·R_sp + ε)
3. Compute abnormal returns (AR) and cumulative abnormal returns (CAR)
   over the event window [t-5, t+22] for all 11 sectors.
4. Classify shocks as demand- vs supply-driven using contemporaneous
   equity co-movement (Kilian & Park 2009 daily analogue).
5. Cross-sectional models: OLS / Ridge / Lasso (original 5 features),
   then extended feature set + Random Forest + Gradient Boosting.
6. Panel model: stack all sectors (N~600) for more stable estimation.
7. Quantile regression on S&P CAR(22d) to characterise tail risk.
8. Leave-One-Out backtest of directional signal → long/short strategy.
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
from collections import OrderedDict
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, LeaveOneOut, GridSearchCV
from sklearn.dummy import DummyRegressor
from scipy import stats
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

try:
    from statsmodels.regression.quantile_regression import QuantReg
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("  [skip] statsmodels not installed — quantile regression disabled")

sns.set_theme(style="darkgrid", context="talk")
plt.rcParams["figure.dpi"] = 120

# ── Parameters ────────────────────────────────────────────────────────────────

ROLL_WINDOW   = 756
MIN_GAP_DAYS  = 22   # no two events within 22 days — prevents overlapping event windows
EST_START     = -270
EST_END       = -30
EVENT_PRE     = 5
EVENT_POST    = 22
CAR_HORIZONS  = [1, 5, 22]

# All SPDR sector ETFs (XLE first for backward compat)
SECTOR_TICKERS = OrderedDict([
    ("xle",  "XLE"),    # Energy              (inception Dec 1998)
    ("xly",  "XLY"),    # Consumer Discret.   (inception Dec 1998)
    ("xli",  "XLI"),    # Industrials         (inception Dec 1998)
    ("xlk",  "XLK"),    # Information Tech    (inception Dec 1998)
    ("xlv",  "XLV"),    # Health Care         (inception Dec 1998)
    ("xlp",  "XLP"),    # Consumer Staples    (inception Dec 1998)
    ("xlf",  "XLF"),    # Financials          (inception Dec 1998)
    ("xlb",  "XLB"),    # Materials           (inception Dec 1998)
    ("xlu",  "XLU"),    # Utilities           (inception Dec 1998)
    ("xlre", "XLRE"),   # Real Estate         (inception Oct 2015)
    ("xlc",  "XLC"),    # Communication Svcs  (inception Jun 2018)
])
ALL_SECTORS  = list(SECTOR_TICKERS.keys())
SECTOR_LABELS = {
    "xle": "Energy",    "xly": "Cons.Disc",  "xli": "Industrials",
    "xlk": "InfoTech",  "xlv": "Healthcare", "xlp": "Cons.Stap",
    "xlf": "Financials","xlb": "Materials",  "xlu": "Utilities",
    "xlre":"RealEstate","xlc": "CommSvc",
}
# Sectors with data from Dec 1998 (full history for PCA)
CORE_SECTORS = [s for s in ALL_SECTORS if s not in ("xlre", "xlc")]

OUTDIR = "plots_event_study"
os.makedirs(OUTDIR, exist_ok=True)

def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, name), bbox_inches="tight")
    plt.close()
    print(f"  saved {OUTDIR}/{name}")

# ── 1. Load Brent crude ───────────────────────────────────────────────────────

oil = pd.read_csv("oil_data.csv")
oil.columns = [c.strip().lower().replace(".", "").replace(" ", "_").replace("%", "pct")
               for c in oil.columns]
oil.columns = [c.lstrip("\ufeff") for c in oil.columns]
oil = oil.rename(columns={"vol": "volume", "change_pct": "raw_chg_pct"})
for col in ["price", "open", "high", "low"]:
    oil[col] = pd.to_numeric(oil[col].astype(str).str.strip(), errors="coerce")
oil["date"]         = pd.to_datetime(oil["date"].astype(str).str.strip(), format="%m/%d/%Y")
oil                 = oil.sort_values("date").reset_index(drop=True)
oil["ret_oil"]      = oil["price"].pct_change()
oil                 = oil.dropna(subset=["ret_oil"]).reset_index(drop=True)
oil["sigma_252"]    = oil["ret_oil"].shift(1).rolling(252).std()
oil["oil_trend_60"] = oil["ret_oil"].shift(1).rolling(60).sum()

# ── 2. Download equity data ───────────────────────────────────────────────────

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
    return raw[["date", f"ret_{col}"]].dropna()

print("Downloading equity data...")
sp = download("^GSPC")

sector_data = {}
for code, ticker in SECTOR_TICKERS.items():
    sector_data[code] = download(ticker)
    print(f"  {ticker} downloaded  ({len(sector_data[code])} rows)")

# VIX
try:
    vix_raw = yf.download("^VIX", start="1988-01-01", end="2026-04-17", progress=False)
    if vix_raw.columns.nlevels > 1:
        vix_raw.columns = vix_raw.columns.get_level_values(0)
    vix_raw = vix_raw.reset_index()
    vix_raw.columns = [c.lower() for c in vix_raw.columns]
    vix_raw["date"] = pd.to_datetime(vix_raw["date"])
    vix_df = vix_raw[["date", "close"]].rename(columns={"close": "vix_close"}).dropna()
    print(f"  VIX downloaded  ({len(vix_df)} rows)")
except Exception:
    vix_df = pd.DataFrame(columns=["date", "vix_close"])
    print("  VIX download failed — vix_pre feature will be NaN")

# Macro controls
def download_level(ticker, col_name):
    try:
        raw = yf.download(ticker, start="1988-01-01", end="2026-04-17", progress=False)
        if raw.empty:
            return pd.DataFrame(columns=["date", col_name])
        if raw.columns.nlevels > 1:
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.reset_index()
        raw.columns = [c.lower() for c in raw.columns]
        raw["date"] = pd.to_datetime(raw["date"])
        return raw[["date", "close"]].rename(columns={"close": col_name}).dropna()
    except Exception:
        return pd.DataFrame(columns=["date", col_name])

lqd_df = download_level("LQD",      "lqd_close")   # IG credit proxy (inception Jul 2002)
dxy_df = download_level("DX-Y.NYB", "dxy_close")   # US Dollar Index
tnx_df = download_level("^TNX",     "tnx_close")   # 10yr Treasury yield
irx_df = download_level("^IRX",     "irx_close")   # 13wk T-bill yield
for label, mdf in [("LQD", lqd_df), ("DXY", dxy_df), ("TNX", tnx_df), ("IRX", irx_df)]:
    if len(mdf) > 0:
        print(f"  {label} downloaded  ({len(mdf)} rows  from {mdf['date'].min().date()})")
    else:
        print(f"  {label} download failed")

# ── 3. Merge ──────────────────────────────────────────────────────────────────

df = oil[["date", "price", "ret_oil", "sigma_252", "oil_trend_60"]].merge(
    sp, on="date", how="inner")
for code, sdf in sector_data.items():
    df = df.merge(sdf[["date", f"ret_{code}"]], on="date", how="left")
df = df.merge(vix_df, on="date", how="left")
for mdf in [lqd_df, dxy_df, tnx_df, irx_df]:
    if len(mdf) > 0:
        df = df.merge(mdf, on="date", how="left")
df = df.sort_values("date").reset_index(drop=True)
# Derived macro columns
if "lqd_close" in df.columns:
    df["ret_lqd"] = df["lqd_close"].pct_change()
if "dxy_close" in df.columns:
    df["ret_dxy"] = df["dxy_close"].pct_change()
if "tnx_close" in df.columns and "irx_close" in df.columns:
    df["term_spread"] = df["tnx_close"] - df["irx_close"]

print(f"\nMerged: {len(df):,} rows  {df['date'].min().date()} → {df['date'].max().date()}")
print("Sector coverage:")
for code in ALL_SECTORS:
    col = f"ret_{code}"
    if col in df.columns:
        first   = df[df[col].notna()]["date"].min()
        n_valid = df[col].notna().sum()
        print(f"  {SECTOR_TICKERS[code]:5s}: from {first.date()}  ({n_valid} trading days)")

# ── 4. Shock identification ───────────────────────────────────────────────────

df["roll_max"]      = df["price"].rolling(ROLL_WINDOW).max()
df["roll_min"]      = df["price"].rolling(ROLL_WINDOW).min()
df["is_3y_high"]    = df["price"] >= df["roll_max"] * 0.999
df["is_3y_low"]     = df["price"] <= df["roll_min"] * 1.001
df["shock_raw"]     = df["is_3y_high"] | df["is_3y_low"]
df["shock_dir"]     = np.where(df["is_3y_high"], 1, np.where(df["is_3y_low"], -1, 0))
df["dist_from_max"] = (df["price"] - df["roll_max"]) / df["roll_max"]

candidates = df.index[df["shock_raw"]].tolist()

def extremeness(i):
    return df.loc[i, "price"] if df.loc[i, "is_3y_high"] else -df.loc[i, "price"]

# Process chronologically; once an event is accepted, suppress the next
# MIN_GAP_DAYS days forward (prevents overlapping event windows).
candidates_sorted = sorted(candidates)   # chronological order
accepted, suppressed = [], set()
for idx in candidates_sorted:
    if idx in suppressed:
        continue
    accepted.append(idx)
    for d in range(1, MIN_GAP_DAYS + 1):   # forward-only suppression
        suppressed.add(idx + d)

df["shock"] = False
df.loc[accepted, "shock"] = True
shock_locs = [df.index.get_loc(i) for i in accepted]

print(f"\nDistinct events : {len(accepted)}")
print(f"  positive (3y high): {df.loc[accepted, 'shock_dir'].eq(1).sum()}")
print(f"  negative (3y low) : {df.loc[accepted, 'shock_dir'].eq(-1).sum()}")
print(f"  pre-2026: {df.loc[accepted, 'date'].dt.year.lt(2026).sum()}")
print(f"  2026    : {df.loc[accepted, 'date'].dt.year.ge(2026).sum()}")

# ── 5. Event study ────────────────────────────────────────────────────────────

records = []

for loc in shock_locs:
    est_s = loc + EST_START
    est_e = loc + EST_END
    ev_s  = loc - EVENT_PRE
    ev_e  = loc + EVENT_POST

    if est_s < 0 or ev_e >= len(df):
        continue

    est = df.iloc[est_s:est_e]
    ev  = df.iloc[ev_s : ev_e + 1].copy()

    if est["ret_gspc"].isna().any() or ev["ret_gspc"].isna().any():
        continue

    t0 = EVENT_PRE

    # S&P 500 baseline: constant mean
    mu_sp = est["ret_gspc"].mean()
    ar_sp = ev["ret_gspc"].values - mu_sp

    # Sector market models: R_sec = α + β·R_sp + ε
    sector_ars   = {}
    sector_betas = {}
    for sec in ALL_SECTORS:
        col = f"ret_{sec}"
        if col not in df.columns:
            continue
        if est[col].isna().any() or ev[col].isna().any():
            continue
        mm = LinearRegression().fit(
            est["ret_gspc"].values.reshape(-1, 1),
            est[col].values)
        sector_ars[sec]   = (ev[col].values
                             - mm.predict(ev["ret_gspc"].values.reshape(-1, 1)))
        sector_betas[sec] = mm.coef_[0]

    # Demand / supply classification (Kilian & Park 2009 daily analogue):
    #   oil and S&P move together  → demand-driven  (+1)
    #   oil and S&P move opposite  → supply-driven  (-1)
    ret_oil_t0  = df.iloc[loc]["ret_oil"]
    ret_gspc_t0 = df.iloc[loc]["ret_gspc"]
    raw_type    = np.sign(ret_oil_t0) * np.sign(ret_gspc_t0)
    shock_type  = int(raw_type) if raw_type != 0 else 1

    # Lagged (pre-determined) demand/supply proxy — fully known before the shock.
    # Uses sign(oil_trend_60) × sign(sp_pre_22): both are pre-shock quantities,
    # so this version is not endogenous to the shock itself.
    oil_t60    = float(df.iloc[loc]["oil_trend_60"])
    sp_p22     = float(df.iloc[loc - 22 : loc]["ret_gspc"].sum())
    raw_lag    = np.sign(oil_t60) * np.sign(sp_p22)
    shock_type_lag = int(raw_lag) if raw_lag != 0 else 1

    # VIX level in 5 days before shock
    vix_slice = df.iloc[max(0, loc - 5) : loc]["vix_close"]
    vix_pre   = float(vix_slice.mean()) if (
        "vix_close" in df.columns and vix_slice.notna().any()) else np.nan

    row = {
        "idx":           loc,
        "date":          df.iloc[loc]["date"],
        "price":         df.iloc[loc]["price"],
        "ret_oil":       ret_oil_t0,
        "ret_oil_abs":   abs(ret_oil_t0),
        "shock_dir":     df.iloc[loc]["shock_dir"],
        "sigma_252":     df.iloc[loc]["sigma_252"],
        "dist_from_max": df.iloc[loc]["dist_from_max"],
        "oil_trend_60":  df.iloc[loc]["oil_trend_60"],
        "shock_type":     shock_type,
        "shock_type_lag": shock_type_lag,
        "vix_pre":       vix_pre,
        "sp_pre_22":     df.iloc[loc - 22 : loc]["ret_gspc"].sum(),
        "sp_pre_5":      df.iloc[loc -  5 : loc]["ret_gspc"].sum(),
        "beta_xle":      sector_betas.get("xle", np.nan),
        "_ar_sp":        ar_sp,
        "_ar_xle":       sector_ars.get("xle", None),
        **{f"_ar_{sec}": sector_ars.get(sec, None) for sec in ALL_SECTORS},
    }

    # CARs: S&P
    for h in CAR_HORIZONS:
        row[f"car_sp_{h}"] = ar_sp[t0 : t0 + h + 1].sum()

    # CARs: all sectors
    for sec in ALL_SECTORS:
        if sec in sector_ars:
            ar = sector_ars[sec]
            for h in CAR_HORIZONS:
                row[f"car_{sec}_{h}"] = ar[t0 : t0 + h + 1].sum()
        else:
            for h in CAR_HORIZONS:
                row[f"car_{sec}_{h}"] = np.nan

    # Macro control features (NaN when series not yet available)
    def _pre5(col):
        if col in df.columns:
            return float(df.iloc[max(0, loc-5):loc][col].sum(min_count=1))
        return np.nan

    row["lqd_pre5"]   = _pre5("ret_lqd")
    row["dxy_pre5"]   = _pre5("ret_dxy")
    row["term_spread"] = (float(df.iloc[loc - 1]["term_spread"])
                          if "term_spread" in df.columns else np.nan)
    row["vix_chg5"]   = (float(df.iloc[loc - 1]["vix_close"] - df.iloc[max(0, loc - 6)]["vix_close"])
                          if "vix_close" in df.columns else np.nan)

    records.append(row)

events = pd.DataFrame(records)
print(f"\nEvents with sufficient data: {len(events)}")
print(f"  pre-2026 : {(events['date'].dt.year < 2026).sum()}")
print(f"  2026     : {(events['date'].dt.year >= 2026).sum()}")

demand_n = int((events[events["date"].dt.year < 2026]["shock_type"] == 1).sum())
supply_n = int((events[events["date"].dt.year < 2026]["shock_type"] ==-1).sum())
print(f"  demand shocks (pre-2026): {demand_n}   supply shocks: {supply_n}")

# ── 6. Cross-sectional model — original 5 features, S&P only ─────────────────

FEATURES = ["shock_dir", "ret_oil", "sigma_252", "sp_pre_22", "dist_from_max"]

ar_arrays = events[["date", "_ar_sp", "_ar_xle", "shock_dir"]].copy()

train = events[events["date"].dt.year < 2026].dropna(
    subset=FEATURES + [f"car_sp_{h}" for h in CAR_HORIZONS]).copy()
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
        target  = f"car_sp_{h}"
        y_train = train[target].values

        m            = make().fit(Xs_train, y_train)
        models[name][h] = m
        y_pred_train = m.predict(Xs_train)
        y_pred_test  = m.predict(Xs_test)
        cv_r2        = cross_val_score(m, Xs_train, y_train, cv=5, scoring="r2").mean()

        y_actual = test[target].values
        mask     = ~np.isnan(y_actual)
        test_mae = mean_absolute_error(y_actual[mask], y_pred_test[mask]) if mask.any() else np.nan

        coef = m.coef_ if hasattr(m, "coef_") else np.full(len(FEATURES), np.nan)
        results.append({
            "model": name, "horizon": h, "cv_r2": cv_r2,
            "train_r2": r2_score(y_train, y_pred_train),
            "test_mae": test_mae, "coefs": dict(zip(FEATURES, coef)),
        })
        test[f"car_sp_{h}_pred_{name}"] = y_pred_test

print("\n=== Cross-sectional model: CAR_SP ~ shock features (original 5) ===")
print(f"  (Null baseline = predict training mean; CV R²=0 by definition)")
print(f"{'Model':6s}  {'H':>3s}  {'cv_R²':>7s}  {'train_R²':>8s}  {'test_MAE':>10s}")
# Null baseline: DummyRegressor(predict mean) — CV R² should be ≈ 0
for h in CAR_HORIZONS:
    null_cv = cross_val_score(DummyRegressor(strategy="mean"),
                              Xs_train, train[f"car_sp_{h}"].values, cv=5, scoring="r2").mean()
    print(f"{'Null':6s}  {h:3d}  {null_cv:7.3f}  {'0.000':>8s}  {'n/a':>10s}")
for r in results:
    mae_s = f"{r['test_mae']:.4f}" if pd.notna(r["test_mae"]) else "    n/a"
    print(f"{r['model']:6s}  {r['horizon']:3d}  {r['cv_r2']:7.3f}  {r['train_r2']:8.3f}  {mae_s:>10s}")

# ── 6b. Extended model: 10 features + RF + GBM ───────────────────────────────

FEATURES_EXT = [
    "shock_dir", "ret_oil", "ret_oil_abs", "sigma_252",
    "sp_pre_22", "sp_pre_5", "dist_from_max",
    "oil_trend_60", "shock_type", "shock_type_lag", "vix_pre",
]

train_ext = train.dropna(
    subset=FEATURES_EXT + [f"car_sp_{h}" for h in CAR_HORIZONS]).copy()
test_ext  = test.dropna(subset=FEATURES_EXT).copy()

scaler_ext   = StandardScaler().fit(train_ext[FEATURES_EXT].values)
Xs_train_ext = scaler_ext.transform(train_ext[FEATURES_EXT].values)
Xs_test_ext  = (scaler_ext.transform(test_ext[FEATURES_EXT].values)
                if len(test_ext) > 0 else np.empty((0, len(FEATURES_EXT))))

EXT_MODEL_SPECS = {
    "Ridge_ext": lambda: RidgeCV(alphas=alphas, cv=5),
    "RF":  lambda: GridSearchCV(
        RandomForestRegressor(n_estimators=300, random_state=42),
        {"max_depth": [2, 3], "min_samples_leaf": [3, 5]},
        cv=5, scoring="r2", refit=True),
    "GBM": lambda: GridSearchCV(
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        {"max_depth": [2, 3], "subsample": [0.8, 1.0]},
        cv=5, scoring="r2", refit=True),
}

models_ext  = {name: {} for name in EXT_MODEL_SPECS}
results_ext = []

for name, make in EXT_MODEL_SPECS.items():
    for h in CAR_HORIZONS:
        target  = f"car_sp_{h}"
        y_train = train_ext[target].values

        m            = make().fit(Xs_train_ext, y_train)
        models_ext[name][h] = m
        y_pred_train = m.predict(Xs_train_ext)

        if hasattr(m, "best_score_"):
            cv_r2 = m.best_score_
        else:
            cv_r2 = cross_val_score(m, Xs_train_ext, y_train, cv=5, scoring="r2").mean()

        if len(test_ext) > 0:
            y_pred_test = m.predict(Xs_test_ext)
            y_actual    = test_ext[target].values
            mask        = ~np.isnan(y_actual)
            test_mae    = mean_absolute_error(y_actual[mask], y_pred_test[mask]) if mask.any() else np.nan
            test_ext[f"car_sp_{h}_pred_{name}"] = y_pred_test
        else:
            test_mae = np.nan

        results_ext.append({
            "model": name, "horizon": h, "cv_r2": cv_r2,
            "train_r2": r2_score(y_train, y_pred_train), "test_mae": test_mae,
        })

print("\n=== Extended model: CAR_SP ~ 11 features (includes RF + GBM) ===")
print(f"{'Model':12s}  {'H':>3s}  {'cv_R²':>7s}  {'train_R²':>8s}  {'test_MAE':>10s}")
for r in results_ext:
    mae_s = f"{r['test_mae']:.4f}" if pd.notna(r["test_mae"]) else "    n/a"
    print(f"{r['model']:12s}  {r['horizon']:3d}  {r['cv_r2']:7.3f}  {r['train_r2']:8.3f}  {mae_s:>10s}")

# ── shock_type endogeneity robustness ─────────────────────────────────────────
# Compare shock_type (contemporaneous, endogenous) vs shock_type_lag
# (pre-determined). Agreement rate and coefficient similarity validate that
# shock_type is not spuriously picking up the shock's own effect.

agree   = (train["shock_type"] == train["shock_type_lag"]).mean()
print(f"\nshock_type vs shock_type_lag agreement rate: {agree:.1%}  (n={len(train)})")

# Coefficients from Ridge_ext at each horizon
st_idx  = FEATURES_EXT.index("shock_type")
stl_idx = FEATURES_EXT.index("shock_type_lag")
print(f"{'Horizon':>8s}  {'shock_type (Ridge)':>20s}  {'shock_type_lag (Ridge)':>23s}")
for r in results_ext:
    if r["model"] != "Ridge_ext":
        continue
    m  = models_ext["Ridge_ext"][r["horizon"]]
    # RidgeCV stores coef_ directly
    coef = m.coef_ if hasattr(m, "coef_") else np.zeros(len(FEATURES_EXT))
    print(f"{r['horizon']:>8d}  {coef[st_idx]:>20.4f}  {coef[stl_idx]:>23.4f}")

# ── 6c. PCA on sector CARs ────────────────────────────────────────────────────

pca_h    = 22
pca_cols = [f"car_{sec}_{pca_h}" for sec in CORE_SECTORS]
pca_src  = train[pca_cols].dropna()
pca_model = None

if len(pca_src) >= 10:
    pca_mat   = pca_src.values
    pca_sc    = StandardScaler().fit_transform(pca_mat)
    pca_model = PCA(n_components=min(len(CORE_SECTORS), 4))
    pca_model.fit(pca_sc)
    scores    = pca_model.transform(pca_sc)
    loadings  = pd.DataFrame(
        pca_model.components_.T,
        index=[SECTOR_LABELS[s] for s in CORE_SECTORS],
        columns=[f"PC{i+1}" for i in range(pca_model.n_components_)])
    pca_evt_idx     = pca_src.index
    pca_shock_types = train.loc[pca_evt_idx, "shock_type"].values
    exp_var         = pca_model.explained_variance_ratio_
    print(f"\nPCA on {pca_h}d sector CARs  (n={len(pca_src)} events, {len(CORE_SECTORS)} sectors)")
    print("  Variance explained: " +
          "  ".join([f"PC{i+1}={v:.1%}" for i, v in enumerate(exp_var)]))
else:
    print(f"\nPCA skipped: only {len(pca_src)} complete rows")

# ── 6d. Panel cross-sectional model ──────────────────────────────────────────

print("\n=== Panel model: all sectors stacked (shock features + sector FE + demand×sector) ===")

panel_rows = []
for _, ev_row in train.iterrows():
    for sec in ALL_SECTORS:
        for h in CAR_HORIZONS:
            car_val = ev_row.get(f"car_{sec}_{h}", np.nan)
            if pd.isna(car_val):
                continue
            base = {f: ev_row.get(f, np.nan) for f in FEATURES_EXT}
            base.update({"sector": sec, "car": car_val, "horizon": h})
            panel_rows.append(base)

panel_df = pd.DataFrame(panel_rows).dropna(subset=FEATURES_EXT + ["car"])

panel_results = []
for h in CAR_HORIZONS:
    sub     = panel_df[panel_df["horizon"] == h].copy()
    sec_dum = pd.get_dummies(sub["sector"], prefix="sec", drop_first=True)
    st_int  = sec_dum.multiply(sub["shock_type"].values, axis=0)
    st_int.columns = [f"st_{c}" for c in st_int.columns]
    Xp  = np.hstack([sub[FEATURES_EXT].values,
                     sec_dum.values.astype(float),
                     st_int.values.astype(float)])
    yp  = sub["car"].values
    sp_sc   = StandardScaler().fit_transform(Xp)
    ridge_p = RidgeCV(alphas=alphas, cv=5).fit(sp_sc, yp)
    lasso_p = LassoCV(alphas=alphas, cv=5, max_iter=10_000).fit(sp_sc, yp)
    cv_r    = cross_val_score(ridge_p, sp_sc, yp, cv=5, scoring="r2").mean()
    cv_l    = cross_val_score(lasso_p, sp_sc, yp, cv=5, scoring="r2").mean()
    panel_results.append({
        "horizon": h, "n_obs": len(sub),
        "ridge_cv_r2": cv_r, "ridge_train_r2": r2_score(yp, ridge_p.predict(sp_sc)),
        "lasso_cv_r2": cv_l,
    })

print(f"{'H':>3s}  {'N':>5s}  {'Ridge_cv_R²':>12s}  {'Ridge_train_R²':>14s}  {'Lasso_cv_R²':>12s}")
for r in panel_results:
    print(f"{r['horizon']:3d}  {r['n_obs']:5d}  {r['ridge_cv_r2']:12.3f}  "
          f"{r['ridge_train_r2']:14.3f}  {r['lasso_cv_r2']:12.3f}")

# ── 6e. Quantile regression ───────────────────────────────────────────────────

h_qr       = 22
qr_results = {}
train_qr   = None

if HAS_STATSMODELS:
    train_qr = train_ext.dropna(
        subset=FEATURES_EXT + [f"car_sp_{h_qr}"]).copy()
    Xq = sm.add_constant(scaler_ext.transform(train_qr[FEATURES_EXT].values))
    yq = train_qr[f"car_sp_{h_qr}"].values
    for tau in [0.10, 0.50, 0.90]:
        qr_results[tau] = QuantReg(yq, Xq).fit(q=tau, max_iter=2000)
    st_idx = FEATURES_EXT.index("shock_type") + 1   # +1 for constant
    print(f"\nQuantile regression  CAR_SP {h_qr}d  (n={len(train_qr)})")
    for tau, res in qr_results.items():
        print(f"  τ={tau:.2f}  intercept={res.params[0]:.4f}  "
              f"shock_type coef={res.params[st_idx]:.4f}")

# ── 6f. Leave-One-Out directional backtest ────────────────────────────────────

h_loo   = 22
y_loo   = train_ext[f"car_sp_{h_loo}"].values

ridge_ext_22  = models_ext["Ridge_ext"][h_loo]
best_alpha_loo = ridge_ext_22.alpha_ if hasattr(ridge_ext_22, "alpha_") else 1.0

loo_preds = np.zeros(len(train_ext))
for tr_idx, te_idx in LeaveOneOut().split(Xs_train_ext):
    m_loo = Ridge(alpha=best_alpha_loo).fit(
        Xs_train_ext[tr_idx], y_loo[tr_idx])
    loo_preds[te_idx] = m_loo.predict(Xs_train_ext[te_idx])

loo_dir_acc      = (np.sign(loo_preds) == np.sign(y_loo)).mean()
loo_strategy_ret = np.sign(loo_preds) * y_loo
loo_benchmark    = y_loo
ann              = np.sqrt(252 / h_loo)

print(f"\nLOO Backtest  (Ridge, S&P CAR {h_loo}d, extended features):")
print(f"  Directional accuracy  : {loo_dir_acc:.1%}")
print(f"  Strategy total CAR    : {loo_strategy_ret.sum()*100:.1f}%")
print(f"  Benchmark total CAR   : {loo_benchmark.sum()*100:.1f}%  (always long)")
print(f"  Strategy Sharpe (ann) : {loo_strategy_ret.mean()/(loo_strategy_ret.std()+1e-10)*ann:.2f}")
print(f"  Benchmark Sharpe (ann): {loo_benchmark.mean()   /(loo_benchmark.std()   +1e-10)*ann:.2f}")

# ── 6g. XLE cross-sectional models ───────────────────────────────────────────
# Mirrors sections 6 / 6b / 6f but targets CAR_XLE instead of CAR_SP.
# Reuses the same scaled feature matrices (Xs_train / Xs_train_ext) since the
# train rows are identical — XLE data covers the full event history.

train_xle = train.dropna(subset=[f"car_xle_{h}" for h in CAR_HORIZONS]).copy()
test_xle  = test.dropna(subset=[f"car_xle_{h}" for h in CAR_HORIZONS] if
                         any(f"car_xle_{h}" in test.columns for h in CAR_HORIZONS)
                         else []).copy()

# Align index positions with Xs_train so we can reuse the scaled matrix
xle_train_mask = train.index.isin(train_xle.index)
Xs_train_xle   = Xs_train[xle_train_mask]
Xs_test_xle    = Xs_test if len(test_xle) > 0 else np.empty((0, len(FEATURES)))

models_xle  = {name: {} for name in MODEL_SPECS}
results_xle = []

for name, make in MODEL_SPECS.items():
    for h in CAR_HORIZONS:
        target  = f"car_xle_{h}"
        y_train = train_xle[target].values
        m = make().fit(Xs_train_xle, y_train)
        models_xle[name][h] = m
        y_pred_train = m.predict(Xs_train_xle)
        cv_r2 = cross_val_score(m, Xs_train_xle, y_train, cv=5, scoring="r2").mean()
        if len(test_xle) > 0:
            y_pred_test = m.predict(Xs_test_xle)
            y_actual    = test_xle[target].values
            mask        = ~np.isnan(y_actual)
            test_mae    = mean_absolute_error(y_actual[mask], y_pred_test[mask]) if mask.any() else np.nan
            test_xle[f"{target}_pred_{name}"] = y_pred_test
        else:
            test_mae = np.nan
        coef = m.coef_ if hasattr(m, "coef_") else np.full(len(FEATURES), np.nan)
        results_xle.append({
            "model": name, "horizon": h, "cv_r2": cv_r2,
            "train_r2": r2_score(y_train, y_pred_train),
            "test_mae": test_mae, "coefs": dict(zip(FEATURES, coef)),
        })

print("\n=== XLE model: CAR_XLE ~ shock features (original 5) ===")
print(f"{'Model':6s}  {'H':>3s}  {'cv_R²':>7s}  {'train_R²':>8s}  {'test_MAE':>10s}")
for h in CAR_HORIZONS:
    null_cv_xle = cross_val_score(DummyRegressor(strategy="mean"),
                                  Xs_train_xle, train_xle[f"car_xle_{h}"].values,
                                  cv=5, scoring="r2").mean()
    print(f"{'Null':6s}  {h:3d}  {null_cv_xle:7.3f}  {'0.000':>8s}  {'n/a':>10s}")
for r in results_xle:
    mae_s = f"{r['test_mae']:.4f}" if pd.notna(r["test_mae"]) else "    n/a"
    print(f"{r['model']:6s}  {r['horizon']:3d}  {r['cv_r2']:7.3f}  {r['train_r2']:8.3f}  {mae_s:>10s}")

# Extended features + RF + GBM for XLE
train_xle_ext = train_ext.dropna(subset=[f"car_xle_{h}" for h in CAR_HORIZONS]).copy()
test_xle_ext  = test_ext.dropna(subset=[f"car_xle_{h}" for h in CAR_HORIZONS] if
                                  any(f"car_xle_{h}" in test_ext.columns for h in CAR_HORIZONS)
                                  else []).copy()

xle_ext_train_mask = train_ext.index.isin(train_xle_ext.index)
Xs_train_xle_ext   = Xs_train_ext[xle_ext_train_mask]
Xs_test_xle_ext    = Xs_test_ext if len(test_xle_ext) > 0 else np.empty((0, len(FEATURES_EXT)))

models_xle_ext  = {name: {} for name in EXT_MODEL_SPECS}
results_xle_ext = []

for name, make in EXT_MODEL_SPECS.items():
    for h in CAR_HORIZONS:
        target  = f"car_xle_{h}"
        y_train = train_xle_ext[target].values
        m = make().fit(Xs_train_xle_ext, y_train)
        models_xle_ext[name][h] = m
        y_pred_train = m.predict(Xs_train_xle_ext)
        cv_r2 = m.best_score_ if hasattr(m, "best_score_") else cross_val_score(
            m, Xs_train_xle_ext, y_train, cv=5, scoring="r2").mean()
        if len(test_xle_ext) > 0:
            y_pred_test = m.predict(Xs_test_xle_ext)
            y_actual    = test_xle_ext[target].values
            mask        = ~np.isnan(y_actual)
            test_mae    = mean_absolute_error(y_actual[mask], y_pred_test[mask]) if mask.any() else np.nan
            test_xle_ext[f"{target}_pred_{name}"] = y_pred_test
        else:
            test_mae = np.nan
        results_xle_ext.append({
            "model": name, "horizon": h, "cv_r2": cv_r2,
            "train_r2": r2_score(y_train, y_pred_train), "test_mae": test_mae,
        })

print("\n=== XLE extended model: CAR_XLE ~ 11 features (RF + GBM) ===")
print(f"{'Model':12s}  {'H':>3s}  {'cv_R²':>7s}  {'train_R²':>8s}  {'test_MAE':>10s}")
for r in results_xle_ext:
    mae_s = f"{r['test_mae']:.4f}" if pd.notna(r["test_mae"]) else "    n/a"
    print(f"{r['model']:12s}  {r['horizon']:3d}  {r['cv_r2']:7.3f}  {r['train_r2']:8.3f}  {mae_s:>10s}")

# LOO backtest for XLE
y_loo_xle      = train_xle_ext[f"car_xle_{h_loo}"].values
best_alpha_xle = models_xle_ext["Ridge_ext"][h_loo].alpha_ if hasattr(
    models_xle_ext["Ridge_ext"][h_loo], "alpha_") else 1.0

loo_preds_xle = np.zeros(len(train_xle_ext))
for tr_idx, te_idx in LeaveOneOut().split(Xs_train_xle_ext):
    m_loo_xle = Ridge(alpha=best_alpha_xle).fit(
        Xs_train_xle_ext[tr_idx], y_loo_xle[tr_idx])
    loo_preds_xle[te_idx] = m_loo_xle.predict(Xs_train_xle_ext[te_idx])

loo_dir_acc_xle      = (np.sign(loo_preds_xle) == np.sign(y_loo_xle)).mean()
loo_strategy_ret_xle = np.sign(loo_preds_xle) * y_loo_xle
loo_benchmark_xle    = y_loo_xle

print(f"\nLOO Backtest  (Ridge, XLE CAR {h_loo}d, extended features):")
print(f"  Directional accuracy  : {loo_dir_acc_xle:.1%}")
print(f"  Strategy total CAR    : {loo_strategy_ret_xle.sum()*100:.1f}%")
print(f"  Benchmark total CAR   : {loo_benchmark_xle.sum()*100:.1f}%  (always long XLE)")
print(f"  Strategy Sharpe (ann) : {loo_strategy_ret_xle.mean()/(loo_strategy_ret_xle.std()+1e-10)*ann:.2f}")
print(f"  Benchmark Sharpe (ann): {loo_benchmark_xle.mean()   /(loo_benchmark_xle.std()   +1e-10)*ann:.2f}")

if len(test_xle) > 0:
    print("\n=== 2026 Events: Predicted vs Actual XLE CAR ===")
    out_xle = test_xle[["date", "ret_oil", "shock_dir"]].copy()
    out_xle["type"] = out_xle["shock_dir"].map({1: "3y-high", -1: "3y-low"})
    for h in CAR_HORIZONS:
        out_xle[f"actual_{h}d"] = (test_xle[f"car_xle_{h}"] * 100).round(2)
        for name in MODEL_SPECS:
            col = f"car_xle_{h}_pred_{name}"
            out_xle[f"{name}_{h}d"] = (test_xle[col] * 100).round(2) if col in test_xle.columns else np.nan
    out_xle = out_xle.drop(columns=["shock_dir"])
    print(out_xle.to_string(index=False))

# ── 6h. Macro-augmented models ────────────────────────────────────────────────
# Adds 4 pre-shock macro controls on top of FEATURES_EXT.
# Run Ridge + Lasso only (honest estimators); compare CV R² to baseline.

MACRO_CONTROLS  = ["dxy_pre5", "term_spread", "vix_chg5"]  # LQD excluded (starts 2002)
FEATURES_MACRO  = FEATURES_EXT + MACRO_CONTROLS

# Events with all macro features available (LQD starts Jul 2002, so some
# early events will be dropped — report n clearly)
train_macro = train.dropna(
    subset=FEATURES_MACRO + [f"car_sp_{h}" for h in CAR_HORIZONS]).copy()
test_macro  = test.dropna(subset=FEATURES_MACRO).copy()

scaler_macro   = StandardScaler().fit(train_macro[FEATURES_MACRO].values)
Xs_train_macro = scaler_macro.transform(train_macro[FEATURES_MACRO].values)
Xs_test_macro  = (scaler_macro.transform(test_macro[FEATURES_MACRO].values)
                  if len(test_macro) > 0 else np.empty((0, len(FEATURES_MACRO))))

print(f"\n=== Macro-augmented models  (n_train={len(train_macro)}, dropped {len(train)-len(train_macro)} events with missing macro data) ===")
print(f"Features: {FEATURES_MACRO}")

MACRO_MODEL_SPECS = {
    "Ridge_macro": lambda: RidgeCV(alphas=alphas, cv=5),
    "Lasso_macro": lambda: LassoCV(alphas=alphas, cv=5, max_iter=10_000),
}

results_macro    = []   # S&P target
results_macro_xle = []  # XLE target

train_macro_xle = train_macro.dropna(
    subset=[f"car_xle_{h}" for h in CAR_HORIZONS]).copy()
xle_macro_mask  = train_macro.index.isin(train_macro_xle.index)
Xs_train_macro_xle = Xs_train_macro[xle_macro_mask]

for name, make in MACRO_MODEL_SPECS.items():
    for asset, results_list, train_df, Xs in [
        ("sp",  results_macro,     train_macro,     Xs_train_macro),
        ("xle", results_macro_xle, train_macro_xle, Xs_train_macro_xle),
    ]:
        for h in CAR_HORIZONS:
            target  = f"car_{asset}_{h}"
            y_train = train_df[target].values
            m       = make().fit(Xs, y_train)
            cv_r2   = cross_val_score(m, Xs, y_train, cv=5, scoring="r2").mean()
            coef    = dict(zip(FEATURES_MACRO, m.coef_))
            results_list.append({
                "model": name, "asset": asset, "horizon": h,
                "cv_r2": cv_r2, "train_r2": r2_score(y_train, m.predict(Xs)),
                "coefs": coef,
            })

# Print comparison table: baseline Ridge_ext vs Ridge_macro
print(f"\n{'Asset':4s}  {'H':>3s}  {'Ridge_ext (10f)':>16s}  {'Ridge_macro (14f)':>18s}  {'Lasso_macro (14f)':>18s}")
baseline_ext = {(r["model"], r["horizon"]): r["cv_r2"]
                for r in results_ext + results_xle_ext
                if r["model"] == "Ridge_ext"}

for asset, res_list, ext_list in [
    ("sp",  results_macro,     results_ext),
    ("xle", results_macro_xle, results_xle_ext),
]:
    for h in CAR_HORIZONS:
        ext_cv  = next((r["cv_r2"] for r in ext_list
                        if r["model"] == "Ridge_ext" and r["horizon"] == h), np.nan)
        ridge_m = next((r["cv_r2"] for r in res_list
                        if r["model"] == "Ridge_macro" and r["horizon"] == h), np.nan)
        lasso_m = next((r["cv_r2"] for r in res_list
                        if r["model"] == "Lasso_macro" and r["horizon"] == h), np.nan)
        print(f"{asset:4s}  {h:3d}  {ext_cv:16.3f}  {ridge_m:18.3f}  {lasso_m:18.3f}")

# Lasso non-zero features (variable selection result)
print("\nLasso_macro non-zero coefficients (XLE 1d — where signal exists):")
xle_1d_lasso = next((r for r in results_macro_xle
                     if r["model"] == "Lasso_macro" and r["horizon"] == 1), None)
if xle_1d_lasso:
    nonzero = {k: v for k, v in xle_1d_lasso["coefs"].items() if abs(v) > 1e-6}
    for k, v in sorted(nonzero.items(), key=lambda x: -abs(x[1])):
        print(f"  {k:20s}  {v:+.4f}")

print("\nLasso_macro non-zero coefficients (S&P 22d):")
sp_22_lasso = next((r for r in results_macro
                    if r["model"] == "Lasso_macro" and r["horizon"] == 22), None)
if sp_22_lasso:
    nonzero = {k: v for k, v in sp_22_lasso["coefs"].items() if abs(v) > 1e-6}
    if nonzero:
        for k, v in sorted(nonzero.items(), key=lambda x: -abs(x[1])):
            print(f"  {k:20s}  {v:+.4f}")
    else:
        print("  (all zeroed out)")

# ── 6i. Secondary shock definition: |ret_oil| > 3σ_252 ───────────────────────

# Identify secondary shocks
df["shock_3sig"] = df["ret_oil"].abs() > 3 * df["sigma_252"]
cands_3sig = sorted(df.index[df["shock_3sig"] & df["sigma_252"].notna()].tolist())

accepted_3sig, supp_3sig = [], set()
for idx in cands_3sig:
    if idx in supp_3sig:
        continue
    accepted_3sig.append(idx)
    for d in range(1, MIN_GAP_DAYS + 1):
        supp_3sig.add(idx + d)

# Direction and dates
df_3sig = pd.DataFrame({
    "idx":       accepted_3sig,
    "date":      df.loc[accepted_3sig, "date"].values,
    "ret_oil":   df.loc[accepted_3sig, "ret_oil"].values,
    "shock_dir": np.sign(df.loc[accepted_3sig, "ret_oil"].values).astype(int),
})

print(f"\n=== Secondary shock definition: |ret_oil| > 3σ_252 ===")
print(f"  Total events    : {len(accepted_3sig)}")
print(f"  Positive (oil↑) : {(df_3sig['shock_dir'] == 1).sum()}")
print(f"  Negative (oil↓) : {(df_3sig['shock_dir']==-1).sum()}")
print(f"  Pre-2026        : {(df_3sig['date'].dt.year < 2026).sum()}")

# Overlap: a primary event overlaps with secondary if there is a secondary
# event on the exact same day OR within ±5 days (same shock episode)
primary_dates  = pd.to_datetime(events["date"].values)
sec_dates      = pd.to_datetime(df_3sig["date"].values)

def overlaps(d, date_set, window=5):
    return any(abs((d - s).days) <= window for s in date_set)

primary_in_sec = sum(overlaps(d, sec_dates) for d in primary_dates)
sec_in_primary = sum(overlaps(d, primary_dates) for d in sec_dates)
sec_only = len(sec_dates) - sec_in_primary

print(f"\nOverlap (±5-day window):")
print(f"  Primary events covered by secondary : {primary_in_sec}/{len(primary_dates)} ({primary_in_sec/len(primary_dates):.0%})")
print(f"  Secondary events overlapping primary : {sec_in_primary}/{len(sec_dates)} ({sec_in_primary/len(sec_dates):.0%})")
print(f"  Secondary-only events               : {sec_only}")

# ── Event study on secondary events ──────────────────────────────────────────

COMPARE_SECTORS = ["sp", "xle", "xly", "xlb", "xlu"]  # S&P + key sectors

def run_event_study(shock_locs, label=""):
    """Compact event study returning CAR DataFrame for COMPARE_SECTORS."""
    recs = []
    for loc in shock_locs:
        est_s, est_e = loc + EST_START, loc + EST_END
        ev_s,  ev_e  = loc - EVENT_PRE,  loc + EVENT_POST
        if est_s < 0 or ev_e >= len(df):
            continue
        est = df.iloc[est_s:est_e]
        ev  = df.iloc[ev_s:ev_e+1].copy()
        if est["ret_gspc"].isna().any() or ev["ret_gspc"].isna().any():
            continue
        t0  = EVENT_PRE
        mu_sp = est["ret_gspc"].mean()
        ar_sp = ev["ret_gspc"].values - mu_sp
        row = {
            "date":      df.iloc[loc]["date"],
            "ret_oil":   df.iloc[loc]["ret_oil"],
            "sigma_252": df.iloc[loc]["sigma_252"],
            "sp_pre_22": df.iloc[loc-22:loc]["ret_gspc"].sum(),
            "dist_from_max": df.iloc[loc]["dist_from_max"],
            "shock_dir": int(np.sign(df.iloc[loc]["ret_oil"])),
            "_ar_sp":    ar_sp,
        }
        for h in CAR_HORIZONS:
            row[f"car_sp_{h}"] = ar_sp[t0:t0+h+1].sum()
        for sec in COMPARE_SECTORS[1:]:
            col = f"ret_{sec}"
            if col not in df.columns or est[col].isna().any() or ev[col].isna().any():
                for h in CAR_HORIZONS:
                    row[f"car_{sec}_{h}"] = np.nan
                row[f"_ar_{sec}"] = None
                continue
            mm    = LinearRegression().fit(est["ret_gspc"].values.reshape(-1,1), est[col].values)
            ar_s  = ev[col].values - mm.predict(ev["ret_gspc"].values.reshape(-1,1))
            row[f"_ar_{sec}"] = ar_s
            for h in CAR_HORIZONS:
                row[f"car_{sec}_{h}"] = ar_s[t0:t0+h+1].sum()
        recs.append(row)
    return pd.DataFrame(recs)

# Run secondary event study (pre-2026 only for comparison)
sec_locs_pre26 = [i for i in accepted_3sig
                  if df.loc[i, "date"].year < 2026]
events_3sig = run_event_study(sec_locs_pre26, "3sig")

print(f"\nSecondary events with sufficient data (pre-2026): {len(events_3sig)}")

# Cross-sectional Ridge on secondary events (original 5 features, S&P + XLE)
FEAT5 = ["shock_dir", "ret_oil", "sigma_252", "sp_pre_22", "dist_from_max"]
print(f"\nRidge CV R² comparison — original 5 features:")
print(f"{'Asset':4s}  {'H':>3s}  {'Primary def':>13s}  {'Secondary def':>14s}")
for asset in ["sp", "xle"]:
    for h in CAR_HORIZONS:
        # primary
        prim_r = next((r["cv_r2"] for r in results if r["model"]=="Ridge" and r["horizon"]==h), np.nan) \
                 if asset=="sp" else \
                 next((r["cv_r2"] for r in results_xle if r["model"]=="Ridge" and r["horizon"]==h), np.nan)
        # secondary
        tgt_col = f"car_{asset}_{h}"
        sec_sub = events_3sig.dropna(subset=FEAT5+[tgt_col])
        if len(sec_sub) >= 10:
            Xs = StandardScaler().fit_transform(sec_sub[FEAT5].values)
            m  = RidgeCV(alphas=alphas, cv=5).fit(Xs, sec_sub[tgt_col].values)
            sec_r = cross_val_score(m, Xs, sec_sub[tgt_col].values, cv=5, scoring="r2").mean()
        else:
            sec_r = np.nan
        print(f"{asset:4s}  {h:3d}  {prim_r:13.3f}  {sec_r:14.3f}")

# Mean CAR comparison table
print(f"\nMean CAR(22d) by sector and definition (pre-2026, positive shocks only):")
print(f"{'Sector':12s}  {'Primary':>10s}  {'Secondary':>11s}")
_train_rows_cmp = events[events["date"].dt.year < 2026]
pos_prim = _train_rows_cmp[_train_rows_cmp["shock_dir"]==1]
pos_sec  = events_3sig[events_3sig["shock_dir"]==1]
for sec in COMPARE_SECTORS:
    col = f"car_{sec}_22"
    pv  = pos_prim[col].dropna().mean()*100 if col in pos_prim.columns else np.nan
    sv  = pos_sec[col].dropna().mean()*100  if col in pos_sec.columns  else np.nan
    lbl = SECTOR_LABELS.get(sec, sec.upper())
    print(f"{lbl:12s}  {pv:10.2f}%  {sv:11.2f}%")

# ── 7. Plots ──────────────────────────────────────────────────────────────────

print("\nSaving plots...")

x  = np.arange(-EVENT_PRE, EVENT_POST + 1)
t0 = EVENT_PRE

def extract_car_paths(subset, ar_col):
    paths = []
    for _, row in subset.iterrows():
        ar = row[ar_col]
        if ar is None:
            continue
        try:
            ar = np.asarray(ar, dtype=float)
        except (TypeError, ValueError):
            continue
        if ar.ndim != 1 or len(ar) != EVENT_PRE + EVENT_POST + 1:
            continue
        car = np.cumsum(ar)
        car = car - car[t0 - 1] if t0 > 0 else car
        paths.append(car)
    return np.array(paths) if paths else np.empty((0, EVENT_PRE + EVENT_POST + 1))

train_rows = events[events["date"].dt.year < 2026]
pos        = train_rows[train_rows["shock_dir"] ==  1]
neg        = train_rows[train_rows["shock_dir"] == -1]

# ── 7a. Average CAR path ──────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
combos = [
    (axes[0, 0], pos, "_ar_sp",  "S&P 500 – Positive shocks",  "darkorange"),
    (axes[0, 1], neg, "_ar_sp",  "S&P 500 – Negative shocks",  "steelblue"),
    (axes[1, 0], pos, "_ar_xle", "XLE – Positive shocks",      "darkorange"),
    (axes[1, 1], neg, "_ar_xle", "XLE – Negative shocks",      "steelblue"),
]
for ax, subset, ar_col, title, color in combos:
    paths = extract_car_paths(subset, ar_col)
    if len(paths) == 0:
        continue
    m, s = paths.mean(0) * 100, paths.std(0) * 100
    se   = s / np.sqrt(len(paths))
    ax.plot(x, m, color=color, linewidth=2, label=f"Mean CAR  (n={len(paths)})")
    ax.fill_between(x, m - 2*se, m + 2*se, alpha=0.35, color=color, label="95% CI (mean)")
    ax.fill_between(x, m - s,   m + s,    alpha=0.12, color=color, label="±1 std")
    ax.axvline(0, linestyle="--", color="black", linewidth=1, label="Shock day")
    ax.axhline(0, linestyle=":",  color="black", linewidth=0.7)
    ax.set_title(title)
    ax.set_xlabel("Days relative to shock")
    ax.set_ylabel("CAR (%)")
    ax.legend(fontsize=9)
plt.suptitle("Event Study: Cumulative Abnormal Returns around Oil Shocks\n"
             "(S&P: constant-mean baseline;  XLE: market model)", fontsize=13)
save("event_study_car.png")

# ── 7b. CAR distributions ─────────────────────────────────────────────────────

fig, axes = plt.subplots(2, len(CAR_HORIZONS), figsize=(5 * len(CAR_HORIZONS), 9))
for col_i, h in enumerate(CAR_HORIZONS):
    for row_i, (asset, color) in enumerate([("sp", "steelblue"), ("xle", "darkorange")]):
        ax       = axes[row_i][col_i]
        col_name = f"car_{asset}_{h}"
        dpos     = train_rows[train_rows["shock_dir"] ==  1][col_name].dropna() * 100
        dneg     = train_rows[train_rows["shock_dir"] == -1][col_name].dropna() * 100
        sns.kdeplot(dpos, ax=ax, color="darkorange", fill=True, alpha=0.35,
                    label=f"+ shock (n={len(dpos)})")
        sns.kdeplot(dneg, ax=ax, color="steelblue",  fill=True, alpha=0.35,
                    label=f"− shock (n={len(dneg)})")
        ax.axvline(dpos.mean(), color="darkorange", linestyle="--", linewidth=1.2)
        ax.axvline(dneg.mean(), color="steelblue",  linestyle="--", linewidth=1.2)
        ax.axvline(0, color="black", linewidth=0.7, linestyle=":")
        _, p_pos = stats.ttest_1samp(dpos, 0)
        _, p_neg = stats.ttest_1samp(dneg, 0)
        ax.set_title(f"{'S&P' if asset=='sp' else 'XLE'} CAR({h}d)\n"
                     rf"$+\mu$={dpos.mean():.2f}% p={p_pos:.2f}  "
                     rf"$-\mu$={dneg.mean():.2f}% p={p_neg:.2f}")
        ax.set_xlabel("CAR (%)")
        ax.legend(fontsize=8)
plt.suptitle("Distribution of Cumulative Abnormal Returns by shock type (pre-2026)")
save("car_distributions.png")

# ── 7c. Scatter: oil return vs S&P CAR ───────────────────────────────────────

fig, axes = plt.subplots(1, len(CAR_HORIZONS), figsize=(5 * len(CAR_HORIZONS), 5))
for ax, h in zip(axes, CAR_HORIZONS):
    xv = train_rows["ret_oil"].values * 100
    yv = train_rows[f"car_sp_{h}"].values * 100
    cs = ["darkorange" if d == 1 else "steelblue" for d in train_rows["shock_dir"]]
    ax.scatter(xv, yv, c=cs, alpha=0.7, s=40, edgecolors="none")
    sl, ic, r, p, _ = stats.linregress(xv, yv)
    xs = np.linspace(xv.min(), xv.max(), 100)
    ax.plot(xs, sl*xs + ic, color="black", linewidth=1.5,
            label=f"OLS  β={sl:.2f}  R²={r**2:.3f}  p={p:.2f}")
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Oil return on shock day (%)")
    ax.set_ylabel(f"S&P CAR(0→{h}d) (%)")
    ax.set_title(f"Oil shock size vs S&P CAR ({h}d)")
    ax.legend(fontsize=9)
plt.suptitle("Cross-sectional: oil shock magnitude → S&P abnormal return")
save("scatter_shock_vs_car.png")

# ── 7d. Coefficient heatmaps (original models) ───────────────────────────────

for name in MODEL_SPECS:
    mres    = [r for r in results if r["model"] == name]
    coef_df = pd.DataFrame({f"{r['horizon']}d": r["coefs"] for r in mres}).T
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(coef_df, annot=True, fmt=".4f", cmap="RdBu_r", center=0, ax=ax,
                linewidths=0.5, cbar_kws={"label": "coef (std features)"})
    ax.set_title(f"{name} – Coefficients for S&P CAR model")
    save(f"coefficients_{name.lower()}.png")

# ── 7e. Model comparison: CV R² ──────────────────────────────────────────────

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
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.legend(title="Model")
plt.suptitle("Model comparison – S&P CAR prediction (original 5 features)")
save("model_comparison.png")

# ── 7f. 2026 predictions vs actuals ──────────────────────────────────────────

if len(test) > 0:
    n_models = len(MODEL_SPECS)
    n_events = len(test)
    fig, axes_ = plt.subplots(len(CAR_HORIZONS), 1,
                              figsize=(max(12, n_events * 3), 5 * len(CAR_HORIZONS)))
    for ax, h in zip(axes_, CAR_HORIZONS):
        target  = f"car_sp_{h}"
        actual  = test[target].values * 100
        dates   = test["date"].values
        dirs    = test["shock_dir"].values
        total_w = 0.75
        bar_w   = total_w / (n_models + 1)
        offsets = np.linspace(-total_w/2, total_w/2, n_models + 1)
        x_pos   = np.arange(n_events)
        has_act = ~np.isnan(actual)
        act_c   = ["darkorange" if d == 1 else "steelblue" for d in dirs]
        if has_act.any():
            ax.bar(x_pos[has_act] + offsets[0], actual[has_act], bar_w,
                   color=[act_c[i] for i in np.where(has_act)[0]],
                   alpha=0.85, label="Actual CAR")
        if (~has_act).any():
            ax.bar(x_pos[~has_act] + offsets[0], np.zeros((~has_act).sum()),
                   bar_w, color="grey", alpha=0.3, hatch="//", label="Pending")
        for i, name in enumerate(MODEL_SPECS):
            pred = test[f"{target}_pred_{name}"].values * 100
            ax.bar(x_pos + offsets[i+1], pred, bar_w,
                   color=MODEL_COLORS[name], alpha=0.75, label=name)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [f"{pd.Timestamp(d).strftime('%b %d')} ({'↑' if di==1 else '↓'})"
             for d, di in zip(dates, dirs)],
            rotation=20, ha="right")
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

# ── 7g. In-sample fit ─────────────────────────────────────────────────────────

dot_c = ["darkorange" if d == 1 else "steelblue" for d in train["shock_dir"]]
fig, axes_ = plt.subplots(len(MODEL_SPECS), len(CAR_HORIZONS),
                          figsize=(5*len(CAR_HORIZONS), 4*len(MODEL_SPECS)), squeeze=False)
for ri, name in enumerate(MODEL_SPECS):
    for ci, h in enumerate(CAR_HORIZONS):
        ax = axes_[ri][ci]
        y  = train[f"car_sp_{h}"].values * 100
        yh = models[name][h].predict(Xs_train) * 100
        r  = next(r for r in results if r["model"] == name and r["horizon"] == h)
        ax.scatter(y, yh, c=dot_c, alpha=0.65, s=30, edgecolors="none")
        lo, hi = min(y.min(), yh.min()), max(y.max(), yh.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel("Actual CAR (%)")
        ax.set_ylabel("Predicted CAR (%)")
        ax.set_title(f"{name} – {h}d  train={r['train_r2']:.3f}  cv={r['cv_r2']:.3f}")
plt.suptitle("In-sample fit (orange=positive shock, blue=negative)", y=1.01)
save("insample_fit.png")

# ── 7h. Sector CAR heatmaps (key new result) ─────────────────────────────────

for shock_d, label_d in [(1, "Positive"), (-1, "Negative")]:
    subset_d = train_rows[train_rows["shock_dir"] == shock_d]
    hm_vals  = pd.DataFrame(index=[SECTOR_LABELS[s] for s in ALL_SECTORS],
                            columns=[f"{h}d" for h in CAR_HORIZONS], dtype=float)
    hm_pval  = pd.DataFrame(index=[SECTOR_LABELS[s] for s in ALL_SECTORS],
                            columns=[f"{h}d" for h in CAR_HORIZONS], dtype=float)
    for sec in ALL_SECTORS:
        lbl = SECTOR_LABELS[sec]
        for h in CAR_HORIZONS:
            vals = subset_d[f"car_{sec}_{h}"].dropna() * 100
            if len(vals) < 3:
                continue
            hm_vals.loc[lbl, f"{h}d"] = vals.mean()
            _, p = stats.ttest_1samp(vals, 0)
            hm_pval.loc[lbl, f"{h}d"] = p

    hm_vals  = hm_vals.sort_values("22d", ascending=False)
    hm_pval  = hm_pval.reindex(hm_vals.index)

    annot = hm_vals.copy().astype(object)
    for idx in hm_vals.index:
        for col in hm_vals.columns:
            v = hm_vals.loc[idx, col]
            p = hm_pval.loc[idx, col]
            if pd.isna(v):
                annot.loc[idx, col] = "n/a"
            else:
                star = ("***" if p < 0.01 else
                        "**"  if p < 0.05 else
                        "*"   if p < 0.1  else "")
                annot.loc[idx, col] = f"{v:.2f}{star}"

    vals_arr = hm_vals.values.astype(float)
    vmax = float(np.nanmax(np.abs(vals_arr))) if not np.all(np.isnan(vals_arr)) else 3.0
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(hm_vals.astype(float), annot=annot, fmt="s",
                cmap="RdYlGn", center=0, vmin=-vmax, vmax=vmax,
                ax=ax, linewidths=0.5, cbar_kws={"label": "Mean CAR (%)"})
    ax.set_title(f"Sector Mean CAR after {label_d} Oil Shocks\n"
                 f"(* p<0.1  ** p<0.05  *** p<0.01;  n={len(subset_d)} events)")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Sector")
    save(f"sector_heatmap_{label_d.lower()}.png")

# ── 7i. Demand vs supply: sector response comparison ─────────────────────────

dem_sub = train_rows[train_rows["shock_type"] ==  1]
sup_sub = train_rows[train_rows["shock_type"] == -1]

dem_means = {}
sup_means = {}
for s in ALL_SECTORS:
    lbl = SECTOR_LABELS[s]
    d_vals = dem_sub[f"car_{s}_22"].dropna()
    s_vals = sup_sub[f"car_{s}_22"].dropna()
    dem_means[lbl] = d_vals.mean() * 100 if len(d_vals) >= 3 else np.nan
    sup_means[lbl] = s_vals.mean() * 100 if len(s_vals) >= 3 else np.nan

order = sorted(dem_means, key=lambda k: dem_means.get(k, 0) or 0)
fig, axes_ = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, means, title in [
    (axes_[0], dem_means, f"Demand shocks (n={len(dem_sub)},  oil↑ + S&P↑)"),
    (axes_[1], sup_means, f"Supply shocks (n={len(sup_sub)},  oil↑ + S&P↓)"),
]:
    vals = [means.get(k, np.nan) for k in order]
    bar_c = ["forestgreen" if (v is not None and not np.isnan(v) and v > 0)
             else "firebrick" for v in vals]
    ax.barh(range(len(order)), vals, color=bar_c, alpha=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean CAR 22d (%)")
    ax.set_title(title)
plt.suptitle("Sector Response by Shock Type: Demand vs Supply\n"
             "(22-day CAR, pre-2026 training events)", fontsize=13)
save("demand_vs_supply_sectors.png")

# ── 7j. PCA: loadings + event scatter ────────────────────────────────────────

if pca_model is not None:
    exp_labels = [f"PC{i+1} ({v:.0%})" for i, v in enumerate(exp_var)]
    loadings.columns = exp_labels
    fig, axes_ = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes_[0]
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, linewidths=0.4, cbar_kws={"label": "loading"})
    ax.set_title("PCA Loadings on Sector CAR(22d)\n(pre-2026, core 9 sectors)")

    ax = axes_[1]
    dm  = pca_shock_types ==  1
    sm_ = pca_shock_types == -1
    ax.scatter(scores[dm,  0], scores[dm,  1], c="darkorange", alpha=0.75,
               s=60, edgecolors="none", label=f"Demand (n={dm.sum()})")
    ax.scatter(scores[sm_, 0], scores[sm_, 1], c="steelblue",  alpha=0.75,
               s=60, edgecolors="none", label=f"Supply (n={sm_.sum()})")
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel(exp_labels[0])
    ax.set_ylabel(exp_labels[1])
    ax.set_title("Events in Principal Component Space\n(demand vs supply)")
    ax.legend()
    plt.suptitle("PCA Decomposition of Cross-Sector Responses to Oil Shocks", fontsize=13)
    save("pca_sectors.png")

# ── 7k. Quantile regression plot ─────────────────────────────────────────────

if HAS_STATSMODELS and qr_results and train_qr is not None:
    fig, ax = plt.subplots(figsize=(9, 6))
    xv = train_qr["ret_oil"].values * 100
    yv = train_qr[f"car_sp_{h_qr}"].values * 100
    cs = ["darkorange" if d == 1 else "steelblue" for d in train_qr["shock_dir"]]
    ax.scatter(xv, yv, c=cs, alpha=0.6, s=40, edgecolors="none",
               label="Events (orange=+shock, blue=−shock)")
    sl, ic, _, _, _ = stats.linregress(xv, yv)
    xs_pct = np.linspace(xv.min(), xv.max(), 100)
    ax.plot(xs_pct, sl*xs_pct + ic, color="black", linewidth=1.8, label="OLS (mean)")

    ret_idx = FEATURES_EXT.index("ret_oil")
    xs_std  = np.linspace(Xs_train_ext[:, ret_idx].min(),
                          Xs_train_ext[:, ret_idx].max(), 100)
    q_colors = {0.10: "firebrick", 0.50: "grey", 0.90: "forestgreen"}
    for tau, res in qr_results.items():
        x_mat = np.zeros((100, len(FEATURES_EXT)))
        x_mat[:, ret_idx] = xs_std
        x_sm  = sm.add_constant(x_mat)
        q_pred = (x_sm @ res.params) * 100
        ax.plot(xs_pct, q_pred, color=q_colors[tau], linewidth=1.5,
                linestyle="--", label=f"QR τ={tau:.0%}")
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Oil return on shock day (%)")
    ax.set_ylabel(f"S&P CAR({h_qr}d) (%)")
    ax.set_title(f"Quantile Regression: S&P CAR({h_qr}d) vs Oil Return\n"
                 f"(other features at mean; captures tail asymmetry)")
    ax.legend(fontsize=9)
    save("quantile_regression.png")

# ── 7l. Feature importance (RF and GBM) ──────────────────────────────────────

for name in ["RF", "GBM"]:
    if name not in models_ext:
        continue
    fig, axes_ = plt.subplots(1, len(CAR_HORIZONS),
                              figsize=(5 * len(CAR_HORIZONS), 5))
    for ax, h in zip(axes_, CAR_HORIZONS):
        m_gs   = models_ext[name][h]
        best_m = m_gs.best_estimator_ if hasattr(m_gs, "best_estimator_") else m_gs
        imp    = best_m.feature_importances_
        feat_s = pd.Series(imp, index=FEATURES_EXT).sort_values(ascending=True)
        feat_s.plot(kind="barh", ax=ax, color="steelblue", alpha=0.8)
        cv_str = f"{m_gs.best_score_:.3f}" if hasattr(m_gs, "best_score_") else "n/a"
        ax.set_title(f"{name} – {h}d  CV R²={cv_str}")
        ax.set_xlabel("Feature importance")
    plt.suptitle(f"{name} Feature Importances — S&P CAR prediction", fontsize=13)
    save(f"feature_importance_{name.lower()}.png")

# ── 7m. LOO strategy PnL ──────────────────────────────────────────────────────

fig, axes_ = plt.subplots(2, 1, figsize=(14, 8))

ax = axes_[0]
cum_strat = np.cumsum(loo_strategy_ret) * 100
cum_bench = np.cumsum(loo_benchmark)    * 100
ax.plot(cum_strat, marker="o", markersize=3, color="crimson",
        label=f"Ridge LOO strategy  (total {cum_strat[-1]:.1f}%,  dir={loo_dir_acc:.0%})")
ax.plot(cum_bench, marker="o", markersize=3, color="steelblue",
        label=f"Always long         (total {cum_bench[-1]:.1f}%)")
ax.axhline(0, color="black", linewidth=0.7)
ax.set_title(f"Leave-One-Out Strategy: S&P CAR {h_loo}d  (Ridge, extended features)")
ax.set_xlabel("Event number (chronological)")
ax.set_ylabel("Cumulative CAR (%)")
ax.legend()

ax = axes_[1]
ax.bar(range(len(loo_strategy_ret)),
       loo_strategy_ret * 100,
       color=["forestgreen" if r >= 0 else "firebrick" for r in loo_strategy_ret],
       alpha=0.75)
ax.axhline(0, color="black", linewidth=0.7)
ax.set_title("Per-event strategy returns (long when model predicts positive CAR)")
ax.set_xlabel("Event number")
ax.set_ylabel("Return (%)")
plt.suptitle("LOO Backtest: Directional Signal → Long/Short S&P on Shock Days", fontsize=12)
save("loo_strategy_pnl.png")

# ── 7n. XLE model comparison ─────────────────────────────────────────────────

perf_xle_df = pd.DataFrame(results_xle)
perf_ext_xle_df = pd.DataFrame(results_xle_ext)

fig, axes_ = plt.subplots(1, 2, figsize=(16, 5))

# Left: S&P vs XLE CV R² side-by-side for Ridge (the honest estimator)
all_ridge = []
for r in results:
    if r["model"] == "Ridge":
        all_ridge.append({"asset": "S&P", "horizon": r["horizon"], "cv_r2": r["cv_r2"]})
for r in results_xle:
    if r["model"] == "Ridge":
        all_ridge.append({"asset": "XLE", "horizon": r["horizon"], "cv_r2": r["cv_r2"]})
ridge_comp = pd.DataFrame(all_ridge).pivot(index="horizon", columns="asset", values="cv_r2")
ridge_comp.plot(kind="bar", ax=axes_[0], rot=0, alpha=0.85,
                color=["navy", "darkorange"])
axes_[0].axhline(0, color="black", linewidth=0.9, linestyle="--")
axes_[0].set_xlabel("Horizon (days)")
axes_[0].set_ylabel("5-fold CV R²")
axes_[0].set_title("Ridge CV R²: S&P vs XLE")
axes_[0].legend(title="Asset")

# Right: XLE extended model CV R²
ext_xle_pivot = perf_ext_xle_df.pivot(index="horizon", columns="model", values="cv_r2")
ext_sp_pivot  = pd.DataFrame(results_ext).pivot(index="horizon", columns="model", values="cv_r2")
combined = ext_xle_pivot.rename(columns=lambda c: f"XLE {c}").join(
           ext_sp_pivot.rename(columns=lambda c: f"S&P {c}"))
combined[["S&P Ridge_ext", "XLE Ridge_ext"]].plot(
    kind="bar", ax=axes_[1], rot=0, alpha=0.85, color=["navy", "darkorange"])
axes_[1].axhline(0, color="black", linewidth=0.9, linestyle="--")
axes_[1].set_xlabel("Horizon (days)")
axes_[1].set_ylabel("5-fold CV R²")
axes_[1].set_title("Ridge_ext CV R²: S&P vs XLE")
axes_[1].legend(title="Asset")

plt.suptitle("S&P vs XLE: Predictability of CAR from Oil Shock Features", fontsize=13)
save("model_comparison_sp_vs_xle.png")

# ── 7o. XLE 2026 predictions ──────────────────────────────────────────────────

if len(test_xle) > 0:
    n_events_xle = len(test_xle)
    fig, axes_ = plt.subplots(len(CAR_HORIZONS), 1,
                              figsize=(max(10, n_events_xle * 4), 5 * len(CAR_HORIZONS)))
    for ax, h in zip(axes_, CAR_HORIZONS):
        target  = f"car_xle_{h}"
        actual  = test_xle[target].values * 100
        dates   = test_xle["date"].values
        dirs    = test_xle["shock_dir"].values
        total_w = 0.75
        bar_w   = total_w / (len(MODEL_SPECS) + 1)
        offsets = np.linspace(-total_w/2, total_w/2, len(MODEL_SPECS) + 1)
        x_pos   = np.arange(n_events_xle)
        has_act = ~np.isnan(actual)
        act_c   = ["darkorange" if d == 1 else "steelblue" for d in dirs]
        if has_act.any():
            ax.bar(x_pos[has_act] + offsets[0], actual[has_act], bar_w,
                   color=[act_c[i] for i in np.where(has_act)[0]], alpha=0.85, label="Actual XLE CAR")
        for i, name in enumerate(MODEL_SPECS):
            col  = f"{target}_pred_{name}"
            pred = test_xle[col].values * 100 if col in test_xle.columns else np.zeros(n_events_xle)
            ax.bar(x_pos + offsets[i+1], pred, bar_w,
                   color=MODEL_COLORS[name], alpha=0.75, label=name)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [f"{pd.Timestamp(d).strftime('%b %d')} ({'↑' if di==1 else '↓'})"
             for d, di in zip(dates, dirs)], rotation=20, ha="right")
        ax.set_ylabel("CAR (%)")
        ax.set_title(f"XLE CAR(0→{h}d) after 2026 oil shocks")
        ax.legend(fontsize=9)
    plt.suptitle("2026 Oil Shock Events: Predicted vs Actual XLE Abnormal Returns", fontsize=13)
    save("predictions_2026_xle.png")

# ── 7p. XLE feature importance ────────────────────────────────────────────────

for name in ["RF", "GBM"]:
    if name not in models_xle_ext:
        continue
    fig, axes_ = plt.subplots(1, len(CAR_HORIZONS),
                              figsize=(5 * len(CAR_HORIZONS), 5))
    for ax, h in zip(axes_, CAR_HORIZONS):
        m_gs   = models_xle_ext[name][h]
        best_m = m_gs.best_estimator_ if hasattr(m_gs, "best_estimator_") else m_gs
        feat_s = pd.Series(best_m.feature_importances_, index=FEATURES_EXT).sort_values(ascending=True)
        feat_s.plot(kind="barh", ax=ax, color="darkorange", alpha=0.8)
        cv_str = f"{m_gs.best_score_:.3f}" if hasattr(m_gs, "best_score_") else "n/a"
        ax.set_title(f"{name} XLE – {h}d  CV R²={cv_str}")
        ax.set_xlabel("Feature importance")
    plt.suptitle(f"{name} Feature Importances — XLE CAR prediction", fontsize=13)
    save(f"feature_importance_{name.lower()}_xle.png")

# ── 7q. XLE LOO strategy PnL ─────────────────────────────────────────────────

fig, axes_ = plt.subplots(2, 1, figsize=(14, 8))

ax = axes_[0]
cum_strat_xle = np.cumsum(loo_strategy_ret_xle) * 100
cum_bench_xle = np.cumsum(loo_benchmark_xle)    * 100
ax.plot(cum_strat_xle, marker="o", markersize=3, color="crimson",
        label=f"Ridge LOO strategy  (total {cum_strat_xle[-1]:.1f}%,  dir={loo_dir_acc_xle:.0%})")
ax.plot(cum_bench_xle, marker="o", markersize=3, color="darkorange",
        label=f"Always long XLE     (total {cum_bench_xle[-1]:.1f}%)")
ax.axhline(0, color="black", linewidth=0.7)
ax.set_title(f"XLE LOO Strategy: CAR {h_loo}d  (Ridge, extended features)")
ax.set_xlabel("Event number (chronological)")
ax.set_ylabel("Cumulative CAR (%)")
ax.legend()

ax = axes_[1]
ax.bar(range(len(loo_strategy_ret_xle)),
       loo_strategy_ret_xle * 100,
       color=["forestgreen" if r >= 0 else "firebrick" for r in loo_strategy_ret_xle],
       alpha=0.75)
ax.axhline(0, color="black", linewidth=0.7)
ax.set_title("Per-event XLE strategy returns")
ax.set_xlabel("Event number")
ax.set_ylabel("Return (%)")
plt.suptitle("XLE LOO Backtest: Directional Signal → Long/Short XLE on Shock Days", fontsize=12)
save("loo_strategy_pnl_xle.png")

# ── 7r. Secondary shock definition: overlap + CAR comparison ─────────────────

# ── 7r-i. Overlap bar chart ───────────────────────────────────────────────────
fig, axes_ = plt.subplots(1, 2, figsize=(14, 5))

ax = axes_[0]
cats   = ["Primary\nonly", "Both\n(±5d)", "Secondary\nonly"]
n_prim_only = len(primary_dates) - primary_in_sec
counts = [n_prim_only, primary_in_sec, sec_only]
colors = ["steelblue", "mediumpurple", "darkorange"]
bars = ax.bar(cats, counts, color=colors, alpha=0.85, edgecolor="white")
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(cnt), ha="center", va="bottom", fontsize=11)
ax.set_ylabel("Number of events")
ax.set_title("Event overlap between definitions\n(primary=3y extreme, secondary=|ret|>3σ)")

ax = axes_[1]
n_pos_sec = int((df_3sig["shock_dir"]==1).sum())
n_neg_sec = int((df_3sig["shock_dir"]==-1).sum())
n_pos_pri = int((events["shock_dir"]==1).sum())
n_neg_pri = int((events["shock_dir"]==-1).sum())
x = np.arange(2)
w = 0.35
ax.bar(x - w/2, [n_pos_pri, n_neg_pri], w, label="Primary",   color="steelblue",  alpha=0.85)
ax.bar(x + w/2, [n_pos_sec, n_neg_sec], w, label="Secondary", color="darkorange", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(["Positive (oil↑)", "Negative (oil↓)"])
ax.set_ylabel("Number of events (pre-2026)")
ax.set_title("Event counts by direction")
ax.legend()
plt.suptitle("Primary vs Secondary Oil Shock Definitions", fontsize=13)
save("secondary_shock_overlap.png")

# ── 7r-ii. CAR path comparison by sector ─────────────────────────────────────

_labels_with_sp = {"sp": "S&P 500", **SECTOR_LABELS}
sector_labels_cmp = {s: _labels_with_sp.get(s, s.upper()) for s in COMPARE_SECTORS}
fig, axes_ = plt.subplots(len(COMPARE_SECTORS), 2,
                          figsize=(16, 4*len(COMPARE_SECTORS)), sharex=True)
x_evt = np.arange(-EVENT_PRE, EVENT_POST + 1)

for row_i, sec in enumerate(COMPARE_SECTORS):
    ar_col = "_ar_sp" if sec == "sp" else f"_ar_{sec}"
    lbl    = sector_labels_cmp[sec]
    for col_i, (subset, def_label, color) in enumerate([
        (train_rows, "Primary (3y extreme)", "steelblue"),
        (events_3sig[events_3sig["date"].dt.year < 2026], "Secondary (|ret|>3σ)", "darkorange"),
    ]):
        ax = axes_[row_i][col_i]
        for shock_d, ls in [(1, "-"), (-1, "--")]:
            sub_d  = subset[subset["shock_dir"] == shock_d]
            paths  = extract_car_paths(sub_d, ar_col)
            if len(paths) == 0:
                continue
            m = paths.mean(0) * 100
            s = paths.std(0)  * 100
            se= s / np.sqrt(len(paths))
            lbl_d = f"{'Pos' if shock_d==1 else 'Neg'} (n={len(paths)})"
            c = "darkorange" if shock_d == 1 else "steelblue"
            ax.plot(x_evt, m, color=c, linewidth=1.8, linestyle=ls, label=lbl_d)
            ax.fill_between(x_evt, m-2*se, m+2*se, alpha=0.25, color=c)
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax.set_title(f"{lbl} — {def_label}")
        ax.set_ylabel("CAR (%)")
        if row_i == 0:
            ax.legend(fontsize=8)
        if row_i == len(COMPARE_SECTORS)-1:
            ax.set_xlabel("Days relative to shock")

plt.suptitle("CAR Paths: Primary vs Secondary Shock Definition\n"
             "(S&P, Energy, Cons.Disc, Materials, Utilities)", fontsize=13)
save("secondary_vs_primary_car.png")

# ── 7r-iii. Mean 22d CAR heatmap comparison ───────────────────────────────────

fig, axes_ = plt.subplots(1, 2, figsize=(12, 5))
for ax, (subset, def_label) in zip(axes_, [
    (train_rows, "Primary (3y extreme)"),
    (events_3sig[events_3sig["date"].dt.year < 2026], "Secondary (|ret|>3σ)"),
]):
    hm = {}
    for shock_d, d_lbl in [(1, "Positive"), (-1, "Negative")]:
        sub_d = subset[subset["shock_dir"] == shock_d]
        hm[d_lbl] = {}
        for sec in COMPARE_SECTORS:
            col = f"car_{sec}_22"
            vals = sub_d[col].dropna() * 100 if col in sub_d.columns else pd.Series(dtype=float)
            hm[d_lbl][sector_labels_cmp[sec]] = vals.mean() if len(vals) >= 3 else np.nan
    hm_df = pd.DataFrame(hm)
    vmax  = max(hm_df.abs().max().max(), 0.1)
    sns.heatmap(hm_df, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                vmin=-vmax, vmax=vmax, ax=ax, linewidths=0.5,
                cbar_kws={"label": "Mean CAR 22d (%)"})
    n = len(subset)
    ax.set_title(f"{def_label}  (n={n})")
plt.suptitle("Mean 22-day Sector CAR: Primary vs Secondary Definition", fontsize=13)
save("secondary_car_heatmap.png")

# ── 7s. Macro controls comparison chart ──────────────────────────────────────

comp_rows = []
for asset, r10_list, r14_list in [
    ("S&P", results_ext,     results_macro),
    ("XLE", results_xle_ext, results_macro_xle),
]:
    for h in CAR_HORIZONS:
        r10 = next((r["cv_r2"] for r in r10_list if r["model"] == "Ridge_ext"   and r["horizon"] == h), np.nan)
        r14 = next((r["cv_r2"] for r in r14_list if r["model"] == "Ridge_macro" and r["horizon"] == h), np.nan)
        l14 = next((r["cv_r2"] for r in r14_list if r["model"] == "Lasso_macro" and r["horizon"] == h), np.nan)
        comp_rows.append({"asset": asset, "horizon": f"{h}d",
                          "Ridge (10f)": r10, "Ridge+macro (14f)": r14, "Lasso+macro (14f)": l14})

comp_df = pd.DataFrame(comp_rows)
fig, axes_ = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
for ax, asset in zip(axes_, ["S&P", "XLE"]):
    sub = comp_df[comp_df["asset"] == asset].set_index("horizon")
    sub[["Ridge (10f)", "Ridge+macro (14f)", "Lasso+macro (14f)"]].plot(
        kind="bar", ax=ax, rot=0, alpha=0.85,
        color=["steelblue", "crimson", "purple"])
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("5-fold CV R²")
    ax.set_title(f"{asset} CAR — baseline vs macro-augmented")
    ax.legend(title="Model", fontsize=8)
    ax.annotate(f"n_macro={len(train_macro)}", xy=(0.02, 0.02),
                xycoords="axes fraction", fontsize=8, color="grey")
plt.suptitle("Effect of Adding Macro Controls (LQD, DXY, Term Spread, VIX Δ)\n"
             "on CV R² — Ridge and Lasso", fontsize=13)
save("macro_controls_comparison.png")

# Lasso non-zero coefficient plot (XLE 1d)
if xle_1d_lasso and xle_1d_lasso["coefs"]:
    fig, ax = plt.subplots(figsize=(10, 5))
    coef_s = pd.Series(xle_1d_lasso["coefs"]).sort_values()
    colors = ["firebrick" if v < 0 else "steelblue" for v in coef_s]
    coef_s.plot(kind="barh", ax=ax, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Lasso_macro: non-zero coefficients for XLE CAR(1d)\n"
                 "(standardised features — shows which macro variables are selected)")
    ax.set_xlabel("Coefficient")
    save("lasso_macro_coefs_xle_1d.png")

# ── 7s. Full history: cumulative returns + shock events ───────────────────────

xle_start_loc = df.loc[df["ret_xle"].notna()].index[0]
sub_hist = df.loc[xle_start_loc:].copy().reset_index(drop=True)
cum_sp_h  = (1 + sub_hist["ret_gspc"]).cumprod() * 100
cum_xle_h = (1 + sub_hist["ret_xle"].fillna(0)).cumprod() * 100
oil_idx_h = sub_hist["price"] / sub_hist["price"].iloc[0] * 100
dates_h   = pd.to_datetime(sub_hist["date"])

fig, axes_ = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
for ax, series, label, color in [
    (axes_[0], cum_sp_h,  "S&P 500 (rebased 100)",    "navy"),
    (axes_[1], cum_xle_h, "XLE  (rebased 100)",        "darkorange"),
    (axes_[2], oil_idx_h, "Brent crude (rebased 100)", "sienna"),
]:
    ax.plot(dates_h, series.values, linewidth=1, color=color)
    ax.axhline(100, color="black", linewidth=0.4, linestyle=":")
    ax.set_ylabel(label, fontsize=10)
    ax.set_yscale("log")

# Vertical lines for each shock event
for _, ev in events.iterrows():
    d = pd.Timestamp(ev["date"])
    c = "darkorange" if ev["shock_dir"] == 1 else "steelblue"
    for ax in axes_:
        ax.axvline(d, color=c, linewidth=0.7, alpha=0.5, linestyle="--")

n_pos = int((events["shock_dir"] == 1).sum())
n_neg = int((events["shock_dir"] ==-1).sum())
pos_p = mpatches.Patch(color="darkorange", alpha=0.7, label=f"Positive shock (n={n_pos})")
neg_p = mpatches.Patch(color="steelblue",  alpha=0.7, label=f"Negative shock (n={n_neg})")
axes_[0].legend(handles=[pos_p, neg_p], fontsize=9, loc="upper left")

axes_[-1].xaxis.set_major_locator(mdates.YearLocator(2))
axes_[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.autofmt_xdate()
plt.suptitle(f"Cumulative Returns of S&P 500, XLE, and Brent Crude\n"
             f"Oil Price Shock Events marked  (n={len(events)} events, log scale)",
             fontsize=13)
save("cumulative_returns_with_events.png")

# ── 7o. 2026 zoom: market prices + model prediction lines ─────────────────────

df_2026     = df[df["date"].dt.year >= 2026].copy().reset_index(drop=True)
events_2026 = events[events["date"].dt.year >= 2026].copy()

if len(df_2026) > 0 and len(events_2026) > 0:
    oil_2026     = df_2026["price"] / df_2026["price"].iloc[0] * 100
    sp_2026_cum  = (1 + df_2026["ret_gspc"].fillna(0)).cumprod() * 100
    xle_2026_cum = (1 + df_2026["ret_xle"].fillna(0)).cumprod() * 100
    dates_2026   = pd.to_datetime(df_2026["date"])

    fig, axes_ = plt.subplots(3, 1, figsize=(14, 13),
                              gridspec_kw={"height_ratios": [1, 1, 1.4]})

    # Panel 1: Oil
    axes_[0].plot(dates_2026, oil_2026, color="sienna", linewidth=1.5)
    axes_[0].axhline(100, color="black", linewidth=0.4, linestyle=":")
    axes_[0].set_ylabel("Brent crude (rebased 100)")

    # Panel 2: S&P500 and XLE
    axes_[1].plot(dates_2026, sp_2026_cum,  color="navy",       linewidth=1.5, label="S&P 500")
    axes_[1].plot(dates_2026, xle_2026_cum, color="darkorange",  linewidth=1.5, label="XLE")
    axes_[1].axhline(100, color="black", linewidth=0.4, linestyle=":")
    axes_[1].set_ylabel("Price index (rebased 100)")
    axes_[1].legend(fontsize=9, loc="lower left")

    # Mark shock events on top two panels
    for _, ev in events_2026.iterrows():
        d   = pd.Timestamp(ev["date"])
        c   = "darkorange" if ev["shock_dir"] == 1 else "steelblue"
        lbl = ev["date"].strftime("%b %d")
        for ax in axes_[:2]:
            ax.axvline(d, color=c, linewidth=2, linestyle="--", alpha=0.85)
            ax.annotate(lbl, xy=(d, 1), xycoords=("data", "axes fraction"),
                        xytext=(4, -4), textcoords="offset points",
                        fontsize=8, color=c, rotation=90, va="top")

    axes_[0].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    axes_[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes_[1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    axes_[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    for ax in axes_[:2]:
        ax.tick_params(axis="x", rotation=30)

    # Panel 3: actual CAR path + model prediction lines per event
    ax_car = axes_[2]
    ax_car.axhline(0, color="black", linewidth=0.9)
    ax_car.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.5)
    ax_car.set_xlabel("Days relative to shock day")
    ax_car.set_ylabel("S&P Cumulative Abnormal Return (%)")

    x_full    = np.arange(-EVENT_PRE, EVENT_POST + 1)
    ev_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    model_styles = [
        ("OLS",   "forestgreen", "--",  "o"),
        ("Ridge", "crimson",     "-.",  "s"),
        ("Lasso", "purple",      ":",   "^"),
    ]

    for i, (_, ev) in enumerate(events_2026.iterrows()):
        ec      = ev_colors[i % len(ev_colors)]
        ev_lbl  = ev["date"].strftime("%b %d")

        # Actual CAR path
        ar = ev.get("_ar_sp")
        if ar is not None:
            try:
                ar_arr   = np.asarray(ar, dtype=float)
                car_path = np.cumsum(ar_arr)
                if t0 > 0:
                    car_path = car_path - car_path[t0 - 1]
                ax_car.plot(x_full, car_path * 100, color=ec, linewidth=2.5,
                            label=f"Actual  {ev_lbl}", zorder=5)
            except Exception:
                pass

        # Model predictions (piecewise lines at 0 → 1d → 5d → 22d)
        test_row = test[test["date"] == ev["date"]]
        if len(test_row) == 0:
            continue
        tr = test_row.iloc[0]
        for name, mc, ms, mk in model_styles:
            pred_x = [0] + list(CAR_HORIZONS)
            pred_y = [0.0]
            for h in CAR_HORIZONS:
                col = f"car_sp_{h}_pred_{name}"
                v   = tr[col] if col in tr.index else np.nan
                pred_y.append(float(v) * 100 if pd.notna(v) else np.nan)
            valid = [(px, py) for px, py in zip(pred_x, pred_y) if not np.isnan(py)]
            if len(valid) > 1:
                pxv, pyv = zip(*valid)
                ax_car.plot(pxv, pyv, color=mc, linewidth=1.6, linestyle=ms,
                            marker=mk, markersize=7, alpha=0.85,
                            label=f"{name}  {ev_lbl}")

    ax_car.legend(fontsize=8, ncol=2, loc="upper left")
    ax_car.set_xlim(-EVENT_PRE - 0.5, EVENT_POST + 0.5)
    ax_car.set_title("S&P 500 CAR path: actual (solid) vs model predictions (dashed)")

    plt.suptitle("2026: Oil Shocks — Market Evolution and Model Predictions", fontsize=13)
    save("market_2026_with_predictions.png")

# ── 7p. 2026 full-sector comparison: expected (training mean) vs actual ───────
# This is the central deliverable of the project — how well did the model
# predict what actually happened across ALL affected markets.

_events_2026 = events[events["date"].dt.year >= 2026].copy()
_train_rows  = events[events["date"].dt.year < 2026]

if len(_events_2026) > 0:
    ev26      = _events_2026.iloc[0]
    sd26      = int(ev26["shock_dir"])
    train_dir = _train_rows[_train_rows["shock_dir"] == sd26]
    dir_lbl   = "positive (3y-high)" if sd26 == 1 else "negative (3y-low)"

    # Build expected vs actual table — S&P first, then all sectors
    _all_compare = [("sp", "S&P 500")] + [(s, SECTOR_LABELS[s]) for s in ALL_SECTORS]
    comp_rows = []
    for sec, lbl in _all_compare:
        for h in CAR_HORIZONS:
            col = f"car_{sec}_{h}"
            exp = train_dir[col].dropna().mean() * 100 if col in train_dir.columns else np.nan
            act = float(ev26[col]) * 100 if (col in ev26.index and pd.notna(ev26.get(col, np.nan))) else np.nan
            comp_rows.append({"sector": lbl, "sec_code": sec,
                               "horizon": h, "expected": exp, "actual": act})
    comp_df = pd.DataFrame(comp_rows)

    # Print table
    print(f"\n=== March 9, 2026 — Expected vs Actual CAR across all sectors ===")
    print(f"Shock type: {dir_lbl}  |  ret_oil = {ev26['ret_oil']*100:.1f}%  |  "
          f"Training mean based on {len(train_dir)} similar events")
    header = f"{'Sector':12s}  {'Exp 1d':>7s}  {'Act 1d':>7s}  {'Exp 5d':>7s}  {'Act 5d':>7s}  {'Exp 22d':>8s}  {'Act 22d':>8s}"
    print(header)
    print("-" * len(header))
    def fmt(v): return f"{v:+7.2f}%" if pd.notna(v) else "    n/a"
    for sec, lbl in _all_compare:
        row = {}
        for h in CAR_HORIZONS:
            sub = comp_df[(comp_df["sec_code"] == sec) & (comp_df["horizon"] == h)]
            row[h] = sub.iloc[0] if len(sub) else {"expected": np.nan, "actual": np.nan}
        sep = "─" * len(header) if sec == "xle" else ""
        if sep: print(sep)
        print(f"{lbl:12s}  {fmt(row[1]['expected'])}  {fmt(row[1]['actual'])}  "
              f"{fmt(row[5]['expected'])}  {fmt(row[5]['actual'])}  "
              f"{fmt(row[22]['expected'])}  {fmt(row[22]['actual'])}")

    # ── Plot: expected vs actual bar chart by sector and horizon ─────────────
    fig, axes_ = plt.subplots(1, len(CAR_HORIZONS), figsize=(20, 7), sharey=False)
    for ax, h in zip(axes_, CAR_HORIZONS):
        sub = comp_df[comp_df["horizon"] == h].copy()
        sub = sub.sort_values("expected", ascending=False).reset_index(drop=True)
        n   = len(sub)
        x   = np.arange(n)
        w   = 0.38
        ax.bar(x - w/2, sub["expected"].fillna(0), w,
               color="steelblue", alpha=0.8, label="Expected\n(training mean)")
        bar_colors = ["forestgreen" if pd.notna(v) and v >= 0 else "firebrick"
                      for v in sub["actual"]]
        ax.bar(x + w/2, sub["actual"].fillna(0), w,
               color=bar_colors, alpha=0.8, label="Actual\n(Mar 9, 2026)")
        # Mark NaN actuals
        for i, v in enumerate(sub["actual"]):
            if pd.isna(v):
                ax.text(x[i] + w/2, 0.1, "n/a", ha="center", va="bottom", fontsize=7, color="grey")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["sector"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("CAR (%)")
        ax.set_title(f"{h}-day CAR")
        if h == CAR_HORIZONS[0]:
            ax.legend(fontsize=9, loc="upper right")

    plt.suptitle(
        f"March 9, 2026 Oil Shock: Expected vs Actual CAR — All 11 Sectors\n"
        f"({dir_lbl}  |  oil return {ev26['ret_oil']*100:.1f}%  |  "
        f"expected = mean of {len(train_dir)} training events with same shock direction)",
        fontsize=12)
    save("2026_sector_comparison.png")

print(f"\nAll plots saved to ./{OUTDIR}/")
