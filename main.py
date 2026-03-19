"""
Seoul Semiconductor Order Data Analysis & 2026 Prediction
=========================================================
Analyzes order history (2021-2025) and predicts monthly delivery quantities
and revenue for 2026 using multiple ML models.

Usage:
    python main.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import Ridge, ElasticNet, Lasso, BayesianRidge
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor, AdaBoostRegressor,
                              StackingRegressor, VotingRegressor)
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
DATA_DIR = 'data'
OUTPUT_DIR = 'output'
CHARTS_DIR = os.path.join(OUTPUT_DIR, 'charts')
DATA_FILES = [f'{year}.xlsx' for year in range(2021, 2026)]

KEY_COLUMNS = [
    'Month', 'Order type', 'Customer name', 'Material name',
    'Delivery qty.', 'Value of supply(KRW)',
    'Application name', 'Sales office name', 'Sales group name',
    'Plant', 'Division name', 'Sales division name',
]

# Chart style
plt.rcParams.update({
    'figure.facecolor': '#0f0f1a',
    'axes.facecolor': '#1a1a2e',
    'axes.edgecolor': '#3a3a5c',
    'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0',
    'xtick.color': '#b0b0cc',
    'ytick.color': '#b0b0cc',
    'grid.color': '#2a2a4a',
    'grid.alpha': 0.6,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.titlesize': 16,
    'legend.facecolor': '#1a1a2e',
    'legend.edgecolor': '#3a3a5c',
})

PALETTE = ['#00d4ff', '#ff6b9d', '#c084fc', '#34d399',
           '#fbbf24', '#f87171', '#60a5fa', '#a78bfa',
           '#fb923c', '#4ade80', '#e879f9', '#22d3ee',
           '#f472b6', '#facc15']


def setup_dirs():
    """Create output directories."""
    os.makedirs(CHARTS_DIR, exist_ok=True)
    print(f"[✓] Output directories ready: {OUTPUT_DIR}/")


# ============================================================
# 1. DATA LOADING & CLEANING
# ============================================================
def load_data():
    """Load all Excel files and return cleaned DataFrame."""
    print("\n" + "=" * 60)
    print("STEP 1: Loading & Cleaning Data")
    print("=" * 60)

    frames = []
    for f in DATA_FILES:
        path = os.path.join(DATA_DIR, f)
        print(f"  Loading {f}...", end=' ')
        df = pd.read_excel(path, usecols=lambda c: c in KEY_COLUMNS)
        print(f"{len(df):,} rows")
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    print(f"\n  Total raw rows: {len(df):,}")

    # Parse Month → datetime
    # Month is stored as float: 2025.01=Jan, 2025.10=Oct
    # Python float drops trailing zero: 2025.10 → 2025.1
    # Use math: extract decimal part × 100 to get month number
    def parse_month_float(val):
        try:
            val = float(val)
            year = int(val)
            decimal_part = val - year  # e.g. 0.01 for Jan, 0.1 for Oct, 0.12 for Dec
            month = round(decimal_part * 100)  # 0.01→1, 0.1→10, 0.12→12
            if month < 1 or month > 12:
                return pd.NaT
            return pd.Timestamp(year=year, month=month, day=1)
        except Exception:
            return pd.NaT

    df['date'] = df['Month'].apply(parse_month_float)
    df = df.dropna(subset=['date'])
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month

    # Remove return / credit orders (negative qty)
    before = len(df)
    df = df[df['Delivery qty.'] > 0].copy()
    print(f"  Removed {before - len(df):,} return/negative-qty rows")

    # Fill NaN strings
    for col in ['Customer name', 'Material name', 'Application name',
                'Sales office name', 'Sales group name', 'Division name']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    df['Value of supply(KRW)'] = pd.to_numeric(
        df['Value of supply(KRW)'], errors='coerce'
    ).fillna(0)

    print(f"  Clean rows: {len(df):,}")
    print(f"  Date range: {df['date'].min().strftime('%Y-%m')} → "
          f"{df['date'].max().strftime('%Y-%m')}")
    return df


# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================
def run_eda(df):
    """Generate EDA charts."""
    print("\n" + "=" * 60)
    print("STEP 2: Exploratory Data Analysis")
    print("=" * 60)

    _chart_monthly_trend(df)
    _chart_top_customers(df)
    _chart_top_materials(df)
    _chart_application_breakdown(df)
    _chart_revenue_trend(df)
    _chart_yoy_heatmap(df)
    print("  [✓] All EDA charts saved")


def _chart_monthly_trend(df):
    monthly = df.groupby('date')['Delivery qty.'].sum().reset_index()
    monthly = monthly.sort_values('date')

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(monthly['date'], monthly['Delivery qty.'],
                    alpha=0.15, color=PALETTE[0])
    ax.plot(monthly['date'], monthly['Delivery qty.'],
            color=PALETTE[0], linewidth=2, marker='o', markersize=3)
    ax.set_title('Monthly Total Delivery Quantity (2021-2025)', fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Delivery Qty')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x / 1e6:.1f}M'))
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '01_monthly_delivery_trend.png'), dpi=150)
    plt.close(fig)
    print("  → 01_monthly_delivery_trend.png")


def _chart_top_customers(df):
    top = (df.groupby('Customer name')['Delivery qty.']
           .sum()
           .nlargest(10)
           .sort_values())

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(top)), top.values, color=PALETTE[:len(top)][::-1],
                   edgecolor='white', linewidth=0.3)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([n[:40] for n in top.index], fontsize=9)
    ax.set_title('Top 10 Customers by Total Delivery Qty', fontweight='bold')
    ax.set_xlabel('Total Delivery Qty')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x / 1e6:.1f}M'))
    for bar, val in zip(bars, top.values):
        ax.text(val, bar.get_y() + bar.get_height() / 2,
                f'  {val / 1e6:.1f}M', va='center', fontsize=9, color='#e0e0e0')
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '02_top10_customers.png'), dpi=150)
    plt.close(fig)
    print("  → 02_top10_customers.png")


def _chart_top_materials(df):
    top = (df.groupby('Material name')['Delivery qty.']
           .sum()
           .nlargest(10)
           .sort_values())

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(top)), top.values, color=PALETTE[:len(top)][::-1],
                   edgecolor='white', linewidth=0.3)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([n[:40] for n in top.index], fontsize=9)
    ax.set_title('Top 10 Materials by Total Delivery Qty', fontweight='bold')
    ax.set_xlabel('Total Delivery Qty')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x / 1e6:.1f}M'))
    for bar, val in zip(bars, top.values):
        ax.text(val, bar.get_y() + bar.get_height() / 2,
                f'  {val / 1e6:.1f}M', va='center', fontsize=9, color='#e0e0e0')
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '03_top10_materials.png'), dpi=150)
    plt.close(fig)
    print("  → 03_top10_materials.png")


def _chart_application_breakdown(df):
    app_data = (df.groupby('Application name')['Delivery qty.']
                .sum()
                .nlargest(8))
    other_val = df['Delivery qty.'].sum() - app_data.sum()
    if other_val > 0:
        app_data = pd.concat([app_data, pd.Series({'Others': other_val})])

    fig, ax = plt.subplots(figsize=(9, 9))
    wedges, texts, autotexts = ax.pie(
        app_data.values, labels=None,
        autopct='%1.1f%%', startangle=140,
        colors=PALETTE[:len(app_data)],
        pctdistance=0.8,
        wedgeprops={'edgecolor': '#0f0f1a', 'linewidth': 1.5}
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_color('#ffffff')
    ax.legend(
        [f'{n[:35]}' for n in app_data.index],
        loc='lower center', bbox_to_anchor=(0.5, -0.08),
        ncol=3, fontsize=8,
        frameon=True, facecolor='#1a1a2e', edgecolor='#3a3a5c'
    )
    ax.set_title('Delivery Qty by Application', fontweight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '04_application_breakdown.png'), dpi=150)
    plt.close(fig)
    print("  → 04_application_breakdown.png")


def _chart_revenue_trend(df):
    monthly = df.groupby('date')['Value of supply(KRW)'].sum().reset_index()
    monthly = monthly.sort_values('date')

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(monthly['date'], monthly['Value of supply(KRW)'],
                    alpha=0.15, color=PALETTE[1])
    ax.plot(monthly['date'], monthly['Value of supply(KRW)'],
            color=PALETTE[1], linewidth=2, marker='o', markersize=3)
    ax.set_title('Monthly Revenue (Value of Supply, KRW)', fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Value of Supply (KRW)')
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x / 1e9:.1f}B'))
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '05_revenue_trend.png'), dpi=150)
    plt.close(fig)
    print("  → 05_revenue_trend.png")


def _chart_yoy_heatmap(df):
    pivot = (df.groupby(['year', 'month_num'])['Delivery qty.']
             .sum()
             .unstack(fill_value=0))

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        pivot / 1e6, annot=True, fmt='.1f', cmap='viridis',
        linewidths=0.5, linecolor='#2a2a4a',
        cbar_kws={'label': 'Delivery Qty (M)'},
        ax=ax
    )
    ax.set_title('Year-over-Year Delivery Qty (Millions)', fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:pivot.shape[1]])
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '06_yoy_heatmap.png'), dpi=150)
    plt.close(fig)
    print("  → 06_yoy_heatmap.png")


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
def build_monthly_features(df):
    """Aggregate to monthly level and engineer features."""
    print("\n" + "=" * 60)
    print("STEP 3: Feature Engineering")
    print("=" * 60)

    monthly = df.groupby('date').agg(
        total_qty=('Delivery qty.', 'sum'),
        total_revenue=('Value of supply(KRW)', 'sum'),
        n_customers=('Customer name', 'nunique'),
        n_materials=('Material name', 'nunique'),
        n_orders=('Delivery qty.', 'count'),
        avg_qty_per_order=('Delivery qty.', 'mean'),
    ).reset_index().sort_values('date')

    monthly['year'] = monthly['date'].dt.year
    monthly['month_num'] = monthly['date'].dt.month

    # Lag features
    for lag in [1, 2, 3, 6, 12]:
        monthly[f'qty_lag_{lag}'] = monthly['total_qty'].shift(lag)
        monthly[f'rev_lag_{lag}'] = monthly['total_revenue'].shift(lag)

    # Rolling averages
    for win in [3, 6, 12]:
        monthly[f'qty_roll_{win}'] = (
            monthly['total_qty'].rolling(win, min_periods=1).mean()
        )
        monthly[f'rev_roll_{win}'] = (
            monthly['total_revenue'].rolling(win, min_periods=1).mean()
        )

    # Year-over-year growth
    monthly['qty_yoy'] = monthly['total_qty'] / monthly['total_qty'].shift(12) - 1

    # Month sin/cos encoding for seasonality
    monthly['month_sin'] = np.sin(2 * np.pi * monthly['month_num'] / 12)
    monthly['month_cos'] = np.cos(2 * np.pi * monthly['month_num'] / 12)

    # Time index (months since start)
    monthly['time_idx'] = np.arange(len(monthly))

    # Interaction / derived features
    monthly['qty_per_customer'] = monthly['total_qty'] / monthly['n_customers'].clip(1)
    monthly['rev_per_qty'] = monthly['total_revenue'] / monthly['total_qty'].clip(1)
    monthly['qty_momentum'] = monthly['total_qty'] - monthly['qty_roll_3']
    monthly['qty_volatility'] = monthly['total_qty'].rolling(6, min_periods=1).std()
    monthly['qty_lag1_x_sin'] = monthly['qty_lag_1'] * monthly['month_sin']
    monthly['qty_lag1_x_cos'] = monthly['qty_lag_1'] * monthly['month_cos']

    monthly = monthly.dropna().reset_index(drop=True)
    print(f"  Monthly feature table: {monthly.shape}")
    print(f"  Features: {[c for c in monthly.columns if c not in ['date']]}")
    return monthly


# ============================================================
# 4. MODEL TRAINING & EVALUATION
# ============================================================
FEATURE_COLS = [
    'month_num', 'year', 'time_idx',
    'qty_lag_1', 'qty_lag_3', 'qty_lag_12',
    'qty_roll_3', 'qty_roll_6', 'qty_roll_12',
    'rev_lag_1', 'rev_lag_12',
    'rev_roll_3', 'rev_roll_12',
    'n_customers', 'n_orders',
    'qty_yoy', 'month_sin', 'month_cos',
    'qty_per_customer', 'rev_per_qty', 'qty_momentum',
    'qty_volatility', 'qty_lag1_x_sin', 'qty_lag1_x_cos',
]


def train_models(monthly):
    """Train multiple models, evaluate, return best + report."""
    print("\n" + "=" * 60)
    print("STEP 4: Model Training & Evaluation")
    print("=" * 60)

    # Train/test split: 2025 as test
    train = monthly[monthly['year'] < 2025].copy()
    test = monthly[monthly['year'] == 2025].copy()

    if len(test) == 0:
        # If 2025 data is incomplete, use last 12 months
        train = monthly.iloc[:-12].copy()
        test = monthly.iloc[-12:].copy()

    X_train = train[FEATURE_COLS]
    y_train = train['total_qty']
    X_test = test[FEATURE_COLS]
    y_test = test['total_qty']

    print(f"  Train: {len(train)} months | Test: {len(test)} months")

    models = {
        'Ridge': Ridge(alpha=500.0),
        'Lasso': Lasso(alpha=50.0, max_iter=5000),
        'ElasticNet': ElasticNet(alpha=50.0, l1_ratio=0.5, max_iter=5000),
        'BayesianRidge': BayesianRidge(),
        'Random Forest': RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        ),
        'Extra Trees': ExtraTreesRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=300, max_depth=3, min_samples_leaf=3,
            learning_rate=0.05, random_state=42
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            min_child_weight=3, reg_alpha=1.0, reg_lambda=5.0,
            random_state=42, verbosity=0
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            min_child_samples=3, reg_alpha=1.0, reg_lambda=5.0,
            random_state=42, verbosity=-1
        ),
    }

    results = {}
    report_lines = []
    report_lines.append("=" * 65)
    report_lines.append("MODEL PERFORMANCE REPORT")
    report_lines.append("=" * 65)
    report_lines.append(f"Train period: {train['date'].min().strftime('%Y-%m')} → "
                        f"{train['date'].max().strftime('%Y-%m')} ({len(train)} months)")
    report_lines.append(f"Test  period: {test['date'].min().strftime('%Y-%m')} → "
                        f"{test['date'].max().strftime('%Y-%m')} ({len(test)} months)")
    report_lines.append("")

    best_model_name = None
    best_r2 = -np.inf

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            'model': model, 'y_pred': y_pred,
            'MAE': mae, 'RMSE': rmse, 'R2': r2
        }

        report_lines.append(f"--- {name} ---")
        report_lines.append(f"  MAE  : {mae:,.0f}")
        report_lines.append(f"  RMSE : {rmse:,.0f}")
        report_lines.append(f"  R²   : {r2:.4f}")
        report_lines.append("")

        print(f"  {name:25s} | MAE={mae:>12,.0f} | RMSE={rmse:>12,.0f} | R²={r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

    # --- Weighted Seasonal Average ---
    try:
        year_weights = {2024: 0.50, 2023: 0.30, 2022: 0.20}
        ws_preds = []
        for _, row in test.iterrows():
            m = int(row['month_num'])
            weighted_sum = 0.0
            weight_total = 0.0
            for yr, w in year_weights.items():
                hist = monthly[(monthly['year'] == yr) & (monthly['month_num'] == m)]
                if len(hist) > 0:
                    weighted_sum += hist['total_qty'].values[0] * w
                    weight_total += w
            if weight_total > 0:
                ws_preds.append(weighted_sum / weight_total)
            else:
                ws_preds.append(row['total_qty'])
        ws_preds = np.array(ws_preds)

        mae = mean_absolute_error(y_test, ws_preds)
        rmse = np.sqrt(mean_squared_error(y_test, ws_preds))
        r2 = r2_score(y_test, ws_preds)

        results['Weighted Seasonal'] = {
            'model': None, 'y_pred': ws_preds,
            'MAE': mae, 'RMSE': rmse, 'R2': r2
        }

        report_lines.append('--- Weighted Seasonal Avg (2024:50%,2023:30%,2022:20%) ---')
        report_lines.append(f'  MAE  : {mae:,.0f}')
        report_lines.append(f'  RMSE : {rmse:,.0f}')
        report_lines.append(f'  R2   : {r2:.4f}')
        report_lines.append('')

        print(f"  {'Weighted Seasonal':25s} | MAE={mae:>12,.0f} | RMSE={rmse:>12,.0f} | R2={r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = 'Weighted Seasonal'
    except Exception as e:
        print(f'  [!] Weighted Seasonal failed: {e}')

    # --- Stacking Ensemble (top 3 ML models) ---
    try:
        sorted_models = sorted(
            [(n, r) for n, r in results.items() if r.get('model') is not None],
            key=lambda x: x[1]['R2'], reverse=True
        )
        top3 = sorted_models[:3]
        if len(top3) >= 2:
            estimators = [(n, type(r['model'])(**r['model'].get_params()))
                          for n, r in top3]
            stacking = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(alpha=10.0),
                cv=3, n_jobs=-1
            )
            stacking.fit(X_train, y_train)
            y_pred = stacking.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            results['Stacking Ensemble'] = {
                'model': stacking, 'y_pred': y_pred,
                'MAE': mae, 'RMSE': rmse, 'R2': r2
            }
            top3_names = '+'.join([n for n, _ in top3])
            report_lines.append(f'--- Stacking Ensemble ({top3_names}) ---')
            report_lines.append(f'  MAE  : {mae:,.0f}')
            report_lines.append(f'  RMSE : {rmse:,.0f}')
            report_lines.append(f'  R2   : {r2:.4f}')
            report_lines.append('')
            print(f"  {'Stacking Ensemble':25s} | MAE={mae:>12,.0f} | RMSE={rmse:>12,.0f} | R2={r2:.4f}")
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = 'Stacking Ensemble'
    except Exception as e:
        print(f'  [!] Stacking Ensemble failed: {e}')

    # SARIMA
    try:
        ts = monthly.set_index('date')['total_qty'].asfreq('MS')
        ts = ts.ffill()
        train_ts = ts[ts.index.year < 2025]
        test_ts = ts[ts.index.year == 2025]
        if len(test_ts) == 0:
            train_ts = ts.iloc[:-12]
            test_ts = ts.iloc[-12:]

        sarima = SARIMAX(train_ts, order=(1, 1, 1),
                         seasonal_order=(1, 1, 1, 12),
                         enforce_stationarity=False,
                         enforce_invertibility=False)
        sarima_fit = sarima.fit(disp=False, maxiter=200)
        n_test = len(test_ts)
        sarima_pred = sarima_fit.forecast(steps=n_test)

        mae = mean_absolute_error(test_ts.values[:n_test], sarima_pred.values[:n_test])
        rmse = np.sqrt(mean_squared_error(test_ts.values[:n_test], sarima_pred.values[:n_test]))
        r2 = r2_score(test_ts.values[:n_test], sarima_pred.values[:n_test])

        results['SARIMA'] = {
            'model': sarima_fit, 'y_pred': sarima_pred.values,
            'MAE': mae, 'RMSE': rmse, 'R2': r2
        }

        report_lines.append(f"--- SARIMA (1,1,1)(1,1,1,12) ---")
        report_lines.append(f"  MAE  : {mae:,.0f}")
        report_lines.append(f"  RMSE : {rmse:,.0f}")
        report_lines.append(f"  R²   : {r2:.4f}")
        report_lines.append("")

        print(f"  {'SARIMA':25s} | MAE={mae:>12,.0f} | RMSE={rmse:>12,.0f} | R²={r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = 'SARIMA'
    except Exception as e:
        print(f"  [!] SARIMA failed: {e}")
        report_lines.append(f"--- SARIMA ---")
        report_lines.append(f"  FAILED: {e}")
        report_lines.append("")

    report_lines.append("=" * 65)
    report_lines.append(f"★ BEST MODEL: {best_model_name} (R² = {best_r2:.4f})")
    report_lines.append("=" * 65)

    # Model comparison chart
    _chart_model_comparison(test, results, monthly)

    # Save report
    report_path = os.path.join(OUTPUT_DIR, 'model_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\n  [✓] Report saved: {report_path}")
    print(f"  ★ Best model: {best_model_name} (R² = {best_r2:.4f})")

    return results, best_model_name, monthly


def _chart_model_comparison(test, results, monthly):
    """Chart comparing actual vs predicted for all models."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot training data
    train = monthly[monthly['date'] < test['date'].min()]
    ax.plot(train['date'], train['total_qty'],
            color='#666688', linewidth=1, alpha=0.5, label='Historical')

    # Actual test
    ax.plot(test['date'], test['total_qty'],
            color='#ffffff', linewidth=2.5, marker='o', markersize=5,
            label='Actual (Test)', zorder=10)

    # Predictions - show only top 5 models by R2
    sorted_res = sorted(results.items(), key=lambda x: x[1]['R2'], reverse=True)
    top_results = sorted_res[:5]
    for i, (name, res) in enumerate(top_results):
        y_pred = res['y_pred'][:len(test)]
        ax.plot(test['date'].values[:len(y_pred)], y_pred,
                color=PALETTE[i], linewidth=2, marker='s', markersize=4,
                linestyle='--', label=f'{name} (R²={res["R2"]:.3f})')

    ax.set_title('Model Comparison: Actual vs Predicted (Test Period)',
                 fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Delivery Qty')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x / 1e6:.1f}M'))
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '07_model_comparison.png'), dpi=150)
    plt.close(fig)
    print("  → 07_model_comparison.png")


# ============================================================
# 5. 2026 PREDICTIONS
# ============================================================
def predict_2026(results, best_model_name, monthly, df):
    """Generate 2026 monthly predictions."""
    print("\n" + "=" * 60)
    print("STEP 5: Generating 2026 Predictions")
    print("=" * 60)

    future_dates = pd.date_range('2026-01-01', periods=12, freq='MS')

    # Use ensemble average of top 3 tree-based models (weighted by R2)
    # Exclude linear models (overfit with many features) and non-retrainable models
    TREE_MODELS = {'Random Forest', 'Extra Trees', 'Gradient Boosting', 'XGBoost', 'LightGBM'}
    ml_results = [(n, r) for n, r in results.items()
                  if r.get('model') is not None
                  and n in TREE_MODELS
                  and r['R2'] < 0.99]  # exclude obvious overfitting
    ml_sorted = sorted(ml_results, key=lambda x: x[1]['R2'], reverse=True)
    top3 = ml_sorted[:3]

    print(f"  Ensemble: averaging top 3 models:")
    ensemble_preds = []
    total_weight = 0
    for name, res in top3:
        r2_w = max(res['R2'], 0.01)  # floor weight at 0.01
        print(f"    - {name} (R2={res['R2']:.4f}, weight={r2_w:.4f})")
        pred_i = _predict_ml_single(res['model'], monthly, future_dates)
        ensemble_preds.append((pred_i, r2_w))
        total_weight += r2_w

    predictions_qty = np.zeros(12)
    for pred_i, w in ensemble_preds:
        predictions_qty += pred_i * (w / total_weight)
    predictions_qty = np.maximum(predictions_qty, 0)

    # Also predict revenue using ML ensemble (same approach as qty)
    predictions_rev = _predict_revenue_ensemble(monthly, future_dates)

    # Build output DataFrame
    pred_df = pd.DataFrame({
        'Month': future_dates,
        'Predicted Delivery Qty': predictions_qty,
        'Predicted Revenue (KRW)': predictions_rev,
    })

    # Add YoY comparison with 2025 (use raw df to include all months, even empty ones)
    raw_2025 = df[df['year'] == 2025].groupby('month_num').agg(
        total_qty=('Delivery qty.', 'sum'),
        total_revenue=('Value of supply(KRW)', 'sum'),
    ).reset_index()
    # Ensure all 12 months exist (fill missing months with 0)
    all_months = pd.DataFrame({'month_num': range(1, 13)})
    raw_2025 = all_months.merge(raw_2025, on='month_num', how='left').fillna(0)
    pred_df['month_num'] = pred_df['Month'].dt.month
    pred_df = pred_df.merge(
        raw_2025.rename(columns={
            'total_qty': '2025 Delivery Qty',
            'total_revenue': '2025 Revenue (KRW)'
        }),
        on='month_num', how='left'
    )
    pred_df['Qty YoY Change (%)'] = np.where(
        pred_df['2025 Delivery Qty'] > 0,
        ((pred_df['Predicted Delivery Qty'] / pred_df['2025 Delivery Qty'] - 1) * 100).round(1),
        0.0
    )
    pred_df['Rev YoY Change (%)'] = np.where(
        pred_df['2025 Revenue (KRW)'] > 0,
        ((pred_df['Predicted Revenue (KRW)'] / pred_df['2025 Revenue (KRW)'] - 1) * 100).round(1),
        0.0
    )
    pred_df = pred_df.drop(columns=['month_num'])

    # Format Month
    pred_df['Month'] = pred_df['Month'].dt.strftime('%Y-%m')

    # Add summary row
    summary = pd.DataFrame([{
        'Month': 'TOTAL 2026',
        'Predicted Delivery Qty': pred_df['Predicted Delivery Qty'].sum(),
        'Predicted Revenue (KRW)': pred_df['Predicted Revenue (KRW)'].sum(),
        '2025 Delivery Qty': pred_df['2025 Delivery Qty'].sum(),
        '2025 Revenue (KRW)': pred_df['2025 Revenue (KRW)'].sum(),
        'Qty YoY Change (%)': (
            (pred_df['Predicted Delivery Qty'].sum() /
             pred_df['2025 Delivery Qty'].sum() - 1) * 100
        ),
        'Rev YoY Change (%)': (
            (pred_df['Predicted Revenue (KRW)'].sum() /
             pred_df['2025 Revenue (KRW)'].sum() - 1) * 100
        ),
    }])
    pred_df = pd.concat([pred_df, summary], ignore_index=True)

    # Save to Excel with formatting
    excel_path = os.path.join(OUTPUT_DIR, 'predictions_2026.xlsx')

    # Customer & Material breakdown predictions
    cust_pred = _predict_customer_breakdown(df, predictions_qty[:12])
    mat_pred = _predict_material_breakdown(df, predictions_qty[:12])

    # Per-customer monthly predictions (Qty + Revenue)
    cust_monthly_qty, cust_monthly_rev = _predict_customer_monthly(
        df, predictions_qty[:12], predictions_rev[:12]
    )

    # Customer × Material monthly predictions
    cust_mat_monthly = _predict_customer_material_monthly(
        df, predictions_qty[:12]
    )

    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        # Write raw data first (will be overwritten with formatting)
        pred_df.to_excel(writer, sheet_name='Monthly Predictions', index=False)
        cust_pred.to_excel(writer, sheet_name='Customer Breakdown', index=False)
        mat_pred.to_excel(writer, sheet_name='Material Breakdown', index=False)
        cust_monthly_qty.to_excel(writer, sheet_name='Customer Monthly Qty', index=False)
        cust_monthly_rev.to_excel(writer, sheet_name='Customer Monthly Rev', index=False)
        cust_mat_monthly.to_excel(writer, sheet_name='Customer-Material Monthly', index=False)

        wb = writer.book

        # ---- Shared formats ----
        fmt_header = wb.add_format({
            'bold': True, 'font_color': '#FFFFFF', 'bg_color': '#1a1a5e',
            'border': 1, 'border_color': '#3a3a8c',
            'text_wrap': True, 'valign': 'vcenter', 'align': 'center',
            'font_size': 11, 'font_name': 'Calibri',
        })
        fmt_num = wb.add_format({
            'num_format': '#,##0', 'font_size': 11, 'font_name': 'Calibri',
            'align': 'right', 'border': 1, 'border_color': '#d0d0d0',
        })
        fmt_num_alt = wb.add_format({
            'num_format': '#,##0', 'font_size': 11, 'font_name': 'Calibri',
            'align': 'right', 'border': 1, 'border_color': '#d0d0d0',
            'bg_color': '#f0f4ff',
        })
        fmt_pct = wb.add_format({
            'num_format': '+0.0%;-0.0%;"0.0%"', 'font_size': 11,
            'font_name': 'Calibri', 'align': 'center',
            'border': 1, 'border_color': '#d0d0d0',
        })
        fmt_pct_alt = wb.add_format({
            'num_format': '+0.0%;-0.0%;"0.0%"', 'font_size': 11,
            'font_name': 'Calibri', 'align': 'center',
            'border': 1, 'border_color': '#d0d0d0',
            'bg_color': '#f0f4ff',
        })
        fmt_month = wb.add_format({
            'font_size': 11, 'font_name': 'Calibri', 'align': 'center',
            'border': 1, 'border_color': '#d0d0d0', 'bold': True,
        })
        fmt_month_alt = wb.add_format({
            'font_size': 11, 'font_name': 'Calibri', 'align': 'center',
            'border': 1, 'border_color': '#d0d0d0', 'bold': True,
            'bg_color': '#f0f4ff',
        })
        fmt_total_label = wb.add_format({
            'bold': True, 'font_size': 12, 'font_name': 'Calibri',
            'align': 'center', 'bg_color': '#fef3c7', 'border': 2,
            'border_color': '#d97706', 'font_color': '#92400e',
        })
        fmt_total_num = wb.add_format({
            'bold': True, 'num_format': '#,##0', 'font_size': 12,
            'font_name': 'Calibri', 'align': 'right',
            'bg_color': '#fef3c7', 'border': 2, 'border_color': '#d97706',
            'font_color': '#92400e',
        })
        fmt_total_pct = wb.add_format({
            'bold': True, 'num_format': '+0.0%;-0.0%;"0.0%"',
            'font_size': 12, 'font_name': 'Calibri', 'align': 'center',
            'bg_color': '#fef3c7', 'border': 2, 'border_color': '#d97706',
            'font_color': '#92400e',
        })
        fmt_text = wb.add_format({
            'font_size': 11, 'font_name': 'Calibri',
            'border': 1, 'border_color': '#d0d0d0',
        })
        fmt_text_alt = wb.add_format({
            'font_size': 11, 'font_name': 'Calibri',
            'border': 1, 'border_color': '#d0d0d0', 'bg_color': '#f0f4ff',
        })
        fmt_pct_plain = wb.add_format({
            'num_format': '0.00"%"', 'font_size': 11,
            'font_name': 'Calibri', 'align': 'center',
            'border': 1, 'border_color': '#d0d0d0',
        })
        fmt_pct_plain_alt = wb.add_format({
            'num_format': '0.00"%"', 'font_size': 11,
            'font_name': 'Calibri', 'align': 'center',
            'border': 1, 'border_color': '#d0d0d0', 'bg_color': '#f0f4ff',
        })
        fmt_total_pct_plain = wb.add_format({
            'bold': True, 'num_format': '0.00"%"',
            'font_size': 12, 'font_name': 'Calibri', 'align': 'center',
            'bg_color': '#fef3c7', 'border': 2, 'border_color': '#d97706',
            'font_color': '#92400e',
        })

        # =============================================
        # Sheet 1: Monthly Predictions
        # =============================================
        ws1 = writer.sheets['Monthly Predictions']
        ws1.freeze_panes(1, 0)
        ws1.set_tab_color('#1a1a5e')

        headers_1 = [
            'Month', 'Predicted\nDelivery Qty', 'Predicted\nRevenue (KRW)',
            '2025\nDelivery Qty', '2025\nRevenue (KRW)',
            'Qty YoY\nChange', 'Rev YoY\nChange',
        ]
        col_widths_1 = [14, 22, 22, 22, 22, 15, 15]
        for c, (h, w) in enumerate(zip(headers_1, col_widths_1)):
            ws1.write(0, c, h, fmt_header)
            ws1.set_column(c, c, w)
        ws1.set_row(0, 36)

        n_data = len(pred_df) - 1  # exclude TOTAL row
        for r in range(n_data):
            is_alt = r % 2 == 1
            row_data = pred_df.iloc[r]
            ws1.write(r + 1, 0, row_data['Month'],
                      fmt_month_alt if is_alt else fmt_month)
            ws1.write_number(r + 1, 1, row_data['Predicted Delivery Qty'],
                             fmt_num_alt if is_alt else fmt_num)
            ws1.write_number(r + 1, 2, row_data['Predicted Revenue (KRW)'],
                             fmt_num_alt if is_alt else fmt_num)
            val_dq = row_data['2025 Delivery Qty']
            val_dr = row_data['2025 Revenue (KRW)']
            if pd.notna(val_dq):
                ws1.write_number(r + 1, 3, val_dq,
                                 fmt_num_alt if is_alt else fmt_num)
            else:
                ws1.write(r + 1, 3, 'N/A',
                          fmt_month_alt if is_alt else fmt_month)
            if pd.notna(val_dr):
                ws1.write_number(r + 1, 4, val_dr,
                                 fmt_num_alt if is_alt else fmt_num)
            else:
                ws1.write(r + 1, 4, 'N/A',
                          fmt_month_alt if is_alt else fmt_month)
            val_qy = row_data['Qty YoY Change (%)']
            val_ry = row_data['Rev YoY Change (%)']
            if pd.notna(val_qy):
                ws1.write_number(r + 1, 5, val_qy / 100,
                                 fmt_pct_alt if is_alt else fmt_pct)
            else:
                ws1.write(r + 1, 5, 'N/A',
                          fmt_month_alt if is_alt else fmt_month)
            if pd.notna(val_ry):
                ws1.write_number(r + 1, 6, val_ry / 100,
                                 fmt_pct_alt if is_alt else fmt_pct)
            else:
                ws1.write(r + 1, 6, 'N/A',
                          fmt_month_alt if is_alt else fmt_month)

        # TOTAL row
        tr = n_data + 1
        total_row = pred_df.iloc[-1]
        ws1.write(tr, 0, 'TOTAL 2026', fmt_total_label)
        ws1.write_number(tr, 1, total_row['Predicted Delivery Qty'], fmt_total_num)
        ws1.write_number(tr, 2, total_row['Predicted Revenue (KRW)'], fmt_total_num)
        ws1.write_number(tr, 3, total_row['2025 Delivery Qty'], fmt_total_num)
        ws1.write_number(tr, 4, total_row['2025 Revenue (KRW)'], fmt_total_num)
        ws1.write_number(tr, 5, total_row['Qty YoY Change (%)'] / 100, fmt_total_pct)
        ws1.write_number(tr, 6, total_row['Rev YoY Change (%)'] / 100, fmt_total_pct)
        ws1.set_row(tr, 24)

        # =============================================
        # Sheet 2: Customer Breakdown
        # =============================================
        ws2 = writer.sheets['Customer Breakdown']
        ws2.freeze_panes(1, 0)
        ws2.set_tab_color('#34d399')

        headers_2 = ['Customer', 'Share', 'Predicted\n2026 Qty']
        col_widths_2 = [42, 12, 22]
        for c, (h, w) in enumerate(zip(headers_2, col_widths_2)):
            ws2.write(0, c, h, fmt_header)
            ws2.set_column(c, c, w)
        ws2.set_row(0, 36)

        for r in range(len(cust_pred)):
            is_alt = r % 2 == 1
            ws2.write(r + 1, 0, cust_pred.iloc[r]['Customer'],
                      fmt_text_alt if is_alt else fmt_text)
            ws2.write_number(r + 1, 1, cust_pred.iloc[r]['Share (%)'],
                             fmt_pct_plain_alt if is_alt else fmt_pct_plain)
            ws2.write_number(r + 1, 2, cust_pred.iloc[r]['Predicted 2026 Qty'],
                             fmt_num_alt if is_alt else fmt_num)

        # Total row for customers
        tr2 = len(cust_pred) + 1
        ws2.write(tr2, 0, 'TOTAL (Top 15)', fmt_total_label)
        ws2.write_number(tr2, 1, cust_pred['Share (%)'].sum(), fmt_total_pct_plain)
        ws2.write_number(tr2, 2, cust_pred['Predicted 2026 Qty'].sum(), fmt_total_num)
        ws2.set_row(tr2, 24)

        # =============================================
        # Sheet 3: Material Breakdown
        # =============================================
        ws3 = writer.sheets['Material Breakdown']
        ws3.freeze_panes(1, 0)
        ws3.set_tab_color('#c084fc')

        headers_3 = ['Material', 'Share', 'Predicted\n2026 Qty']
        col_widths_3 = [32, 12, 22]
        for c, (h, w) in enumerate(zip(headers_3, col_widths_3)):
            ws3.write(0, c, h, fmt_header)
            ws3.set_column(c, c, w)
        ws3.set_row(0, 36)

        for r in range(len(mat_pred)):
            is_alt = r % 2 == 1
            ws3.write(r + 1, 0, mat_pred.iloc[r]['Material'],
                      fmt_text_alt if is_alt else fmt_text)
            ws3.write_number(r + 1, 1, mat_pred.iloc[r]['Share (%)'],
                             fmt_pct_plain_alt if is_alt else fmt_pct_plain)
            ws3.write_number(r + 1, 2, mat_pred.iloc[r]['Predicted 2026 Qty'],
                             fmt_num_alt if is_alt else fmt_num)

        # Total row for materials
        tr3 = len(mat_pred) + 1
        ws3.write(tr3, 0, 'TOTAL (Top 15)', fmt_total_label)
        ws3.write_number(tr3, 1, mat_pred['Share (%)'].sum(), fmt_total_pct_plain)
        ws3.write_number(tr3, 2, mat_pred['Predicted 2026 Qty'].sum(), fmt_total_num)
        ws3.set_row(tr3, 24)

        # =============================================
        # Sheet 4 & 5: Customer Monthly Qty / Rev
        # =============================================
        for sheet_name, cust_df, tab_color in [
            ('Customer Monthly Qty', cust_monthly_qty, '#34d399'),
            ('Customer Monthly Rev', cust_monthly_rev, '#fbbf24'),
        ]:
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 1)  # freeze header + customer column
            ws.set_tab_color(tab_color)

            cols = list(cust_df.columns)
            # Write headers
            for c, col_name in enumerate(cols):
                ws.write(0, c, col_name, fmt_header)
                if c == 0:
                    ws.set_column(c, c, 28)  # Customer column
                else:
                    ws.set_column(c, c, 16)  # Month/Total columns
            ws.set_row(0, 30)

            # Write data rows
            for r in range(len(cust_df)):
                is_alt = r % 2 == 1
                ws.write(r + 1, 0, cust_df.iloc[r][cols[0]],
                         fmt_text_alt if is_alt else fmt_text)
                for c in range(1, len(cols)):
                    val = cust_df.iloc[r][cols[c]]
                    ws.write_number(r + 1, c, float(val),
                                    fmt_num_alt if is_alt else fmt_num)

            # Total row
            tr = len(cust_df) + 1
            ws.write(tr, 0, 'TOTAL (Top 15)', fmt_total_label)
            for c in range(1, len(cols)):
                col_sum = cust_df[cols[c]].sum()
                ws.write_number(tr, c, float(col_sum), fmt_total_num)
            ws.set_row(tr, 24)

        # =============================================
        # Sheet 6: Customer-Material Monthly
        # =============================================
        ws6 = writer.sheets['Customer-Material Monthly']
        ws6.freeze_panes(1, 2)  # freeze header + Customer/Material cols
        ws6.set_tab_color('#ff6b9d')

        cm_cols = list(cust_mat_monthly.columns)
        for c, col_name in enumerate(cm_cols):
            ws6.write(0, c, col_name, fmt_header)
            if c == 0:
                ws6.set_column(c, c, 30)  # Customer
            elif c == 1:
                ws6.set_column(c, c, 28)  # Material
            else:
                ws6.set_column(c, c, 14)  # Month/Total
        ws6.set_row(0, 30)

        for r in range(len(cust_mat_monthly)):
            is_alt = r % 2 == 1
            ws6.write(r + 1, 0, str(cust_mat_monthly.iloc[r][cm_cols[0]]),
                      fmt_text_alt if is_alt else fmt_text)
            ws6.write(r + 1, 1, str(cust_mat_monthly.iloc[r][cm_cols[1]]),
                      fmt_text_alt if is_alt else fmt_text)
            for c in range(2, len(cm_cols)):
                val = cust_mat_monthly.iloc[r][cm_cols[c]]
                ws6.write_number(r + 1, c, float(val),
                                fmt_num_alt if is_alt else fmt_num)

        # Total row
        tr6 = len(cust_mat_monthly) + 1
        ws6.write(tr6, 0, 'TOTAL', fmt_total_label)
        ws6.write(tr6, 1, '', fmt_total_label)
        for c in range(2, len(cm_cols)):
            col_sum = cust_mat_monthly[cm_cols[c]].sum()
            ws6.write_number(tr6, c, float(col_sum), fmt_total_num)
        ws6.set_row(tr6, 24)

    print(f"  [OK] Predictions saved: {excel_path}")

    # Prediction chart
    _chart_predictions(monthly, future_dates, predictions_qty, predictions_rev, best_model_name)
    _chart_customer_forecast(df, predictions_qty)
    _chart_customer_material_heatmap(cust_mat_monthly)

    return pred_df


def _predict_sarima(results, monthly):
    """Forecast 2026 with SARIMA retrained on full data."""
    ts = monthly.set_index('date')['total_qty'].asfreq('MS').ffill()
    sarima = SARIMAX(ts, order=(1, 1, 1),
                     seasonal_order=(1, 1, 1, 12),
                     enforce_stationarity=False,
                     enforce_invertibility=False)
    fit = sarima.fit(disp=False, maxiter=200)
    forecast = fit.forecast(steps=12)
    return forecast.values


def _predict_weighted_seasonal(monthly):
    """Forecast 2026 using weighted seasonal average of recent years."""
    # Weights: most recent years weighted more heavily
    year_weights = {2025: 0.40, 2024: 0.30, 2023: 0.20, 2022: 0.10}
    preds = []
    for m in range(1, 13):
        weighted_sum = 0.0
        weight_total = 0.0
        for yr, w in year_weights.items():
            hist = monthly[(monthly['year'] == yr) & (monthly['month_num'] == m)]
            if len(hist) > 0:
                weighted_sum += hist['total_qty'].values[0] * w
                weight_total += w
        if weight_total > 0:
            preds.append(weighted_sum / weight_total)
        else:
            # Fallback: use overall average for that month
            m_data = monthly[monthly['month_num'] == m]
            preds.append(m_data['total_qty'].mean())
    return np.array(preds)

def _predict_ml_single(trained_model, monthly, future_dates):
    """Forecast 2026 with a specific model retrained on full data."""
    model_class = type(trained_model)
    params = trained_model.get_params()
    model = model_class(**params)
    model.fit(monthly[FEATURE_COLS], monthly['total_qty'])

    preds = []
    extended = monthly.copy()

    for i, dt in enumerate(future_dates):
        row = {}
        row['date'] = dt
        row['month_num'] = dt.month
        row['year'] = dt.year
        row['time_idx'] = len(extended)

        last_vals = extended['total_qty'].values
        row['qty_lag_1'] = last_vals[-1]
        row['qty_lag_3'] = last_vals[-3]
        row['qty_lag_12'] = last_vals[-12]
        row['qty_roll_3'] = np.mean(last_vals[-3:])
        row['qty_roll_6'] = np.mean(last_vals[-6:])
        row['qty_roll_12'] = np.mean(last_vals[-12:])

        rev_vals = extended['total_revenue'].values
        row['rev_lag_1'] = rev_vals[-1]
        row['rev_lag_12'] = rev_vals[-12]
        row['rev_roll_3'] = np.mean(rev_vals[-3:])
        row['rev_roll_12'] = np.mean(rev_vals[-12:])

        row['n_customers'] = extended['n_customers'].iloc[-12:].mean()
        row['n_orders'] = extended['n_orders'].iloc[-12:].mean()
        row['qty_yoy'] = (last_vals[-1] / last_vals[-12]) - 1
        row['month_sin'] = np.sin(2 * np.pi * dt.month / 12)
        row['month_cos'] = np.cos(2 * np.pi * dt.month / 12)
        row['qty_per_customer'] = last_vals[-1] / max(row['n_customers'], 1)
        row['rev_per_qty'] = rev_vals[-1] / max(last_vals[-1], 1)
        row['qty_momentum'] = last_vals[-1] - row['qty_roll_3']
        row['qty_volatility'] = np.std(last_vals[-6:])
        row['qty_lag1_x_sin'] = row['qty_lag_1'] * row['month_sin']
        row['qty_lag1_x_cos'] = row['qty_lag_1'] * row['month_cos']
        row['total_qty'] = 0
        row['total_revenue'] = rev_vals[-12]

        X_future = pd.DataFrame([row])[FEATURE_COLS]
        pred = max(model.predict(X_future)[0], 0)
        preds.append(pred)

        row['total_qty'] = pred
        extended = pd.concat([extended, pd.DataFrame([row])], ignore_index=True)

    return np.array(preds)


def _predict_ml(results, best_model_name, monthly, future_dates):
    """Forecast 2026 with the best ML model retrained on full data."""
    # Retrain on all data
    model_class = type(results[best_model_name]['model'])
    params = results[best_model_name]['model'].get_params()
    model = model_class(**params)
    model.fit(monthly[FEATURE_COLS], monthly['total_qty'])

    # Build future features step by step (autoregressive)
    preds = []
    extended = monthly.copy()

    for i, dt in enumerate(future_dates):
        row = {}
        row['date'] = dt
        row['month_num'] = dt.month
        row['year'] = dt.year
        row['time_idx'] = len(extended)

        last_vals = extended['total_qty'].values
        row['qty_lag_1'] = last_vals[-1]
        row['qty_lag_3'] = last_vals[-3]
        row['qty_lag_12'] = last_vals[-12]
        row['qty_roll_3'] = np.mean(last_vals[-3:])
        row['qty_roll_6'] = np.mean(last_vals[-6:])
        row['qty_roll_12'] = np.mean(last_vals[-12:])

        rev_vals = extended['total_revenue'].values
        row['rev_lag_1'] = rev_vals[-1]
        row['rev_lag_12'] = rev_vals[-12]
        row['rev_roll_3'] = np.mean(rev_vals[-3:])
        row['rev_roll_12'] = np.mean(rev_vals[-12:])

        row['n_customers'] = extended['n_customers'].iloc[-12:].mean()
        row['n_orders'] = extended['n_orders'].iloc[-12:].mean()
        row['qty_yoy'] = (last_vals[-1] / last_vals[-12]) - 1
        row['month_sin'] = np.sin(2 * np.pi * dt.month / 12)
        row['month_cos'] = np.cos(2 * np.pi * dt.month / 12)
        row['qty_per_customer'] = last_vals[-1] / max(row['n_customers'], 1)
        row['rev_per_qty'] = rev_vals[-1] / max(last_vals[-1], 1)
        row['qty_momentum'] = last_vals[-1] - row['qty_roll_3']
        row['qty_volatility'] = np.std(last_vals[-6:])
        row['qty_lag1_x_sin'] = row['qty_lag_1'] * row['month_sin']
        row['qty_lag1_x_cos'] = row['qty_lag_1'] * row['month_cos']
        row['total_qty'] = 0
        row['total_revenue'] = rev_vals[-12]  # approximate

        X_future = pd.DataFrame([row])[FEATURE_COLS]
        pred = model.predict(X_future)[0]
        pred = max(pred, 0)  # no negative quantities
        preds.append(pred)

        row['total_qty'] = pred
        extended = pd.concat([extended, pd.DataFrame([row])], ignore_index=True)

    return np.array(preds)


def _predict_revenue_ensemble(monthly, future_dates):
    """Forecast 2026 revenue using ensemble of top 3 tree-based models."""
    # Revenue feature columns (reuse FEATURE_COLS which include rev lags)
    target = 'total_revenue'

    # Train same tree-based models for revenue
    rev_models = {
        'RF_rev': RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=3,
            random_state=42, n_jobs=-1),
        'ET_rev': ExtraTreesRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=3,
            random_state=42, n_jobs=-1),
        'XGB_rev': xgb.XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            min_child_weight=3, reg_alpha=1.0, reg_lambda=5.0,
            random_state=42, verbosity=0),
    }

    # Train/test split same as qty
    train = monthly[monthly['year'] < 2025]
    test = monthly[monthly['year'] == 2025]
    if len(test) == 0:
        train = monthly.iloc[:-12]
        test = monthly.iloc[-12:]

    X_train, y_train = train[FEATURE_COLS], train[target]
    X_test, y_test = test[FEATURE_COLS], test[target]

    # Evaluate and pick top 3
    scored = []
    for name, model in rev_models.items():
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        scored.append((name, model, r2))
    scored.sort(key=lambda x: x[2], reverse=True)

    # Ensemble predict using full data retrain
    all_preds = []
    total_w = 0
    for name, trained_model, r2 in scored[:3]:
        w = max(r2, 0.01)
        pred_i = _predict_rev_single(trained_model, monthly, future_dates)
        all_preds.append((pred_i, w))
        total_w += w

    result = np.zeros(12)
    for pred_i, w in all_preds:
        result += pred_i * (w / total_w)
    return np.maximum(result, 0)


def _predict_rev_single(trained_model, monthly, future_dates):
    """Forecast revenue for 12 months with a single model retrained on full data."""
    model_class = type(trained_model)
    params = trained_model.get_params()
    model = model_class(**params)
    model.fit(monthly[FEATURE_COLS], monthly['total_revenue'])

    preds = []
    extended = monthly.copy()

    for dt in future_dates:
        row = {}
        row['date'] = dt
        row['month_num'] = dt.month
        row['year'] = dt.year
        row['time_idx'] = len(extended)

        last_vals = extended['total_qty'].values
        row['qty_lag_1'] = last_vals[-1]
        row['qty_lag_3'] = last_vals[-3]
        row['qty_lag_12'] = last_vals[-12]
        row['qty_roll_3'] = np.mean(last_vals[-3:])
        row['qty_roll_6'] = np.mean(last_vals[-6:])
        row['qty_roll_12'] = np.mean(last_vals[-12:])

        rev_vals = extended['total_revenue'].values
        row['rev_lag_1'] = rev_vals[-1]
        row['rev_lag_12'] = rev_vals[-12]
        row['rev_roll_3'] = np.mean(rev_vals[-3:])
        row['rev_roll_12'] = np.mean(rev_vals[-12:])

        row['n_customers'] = extended['n_customers'].iloc[-12:].mean()
        row['n_orders'] = extended['n_orders'].iloc[-12:].mean()
        row['qty_yoy'] = (last_vals[-1] / last_vals[-12]) - 1
        row['month_sin'] = np.sin(2 * np.pi * dt.month / 12)
        row['month_cos'] = np.cos(2 * np.pi * dt.month / 12)
        row['qty_per_customer'] = last_vals[-1] / max(row['n_customers'], 1)
        row['rev_per_qty'] = rev_vals[-1] / max(last_vals[-1], 1)
        row['qty_momentum'] = last_vals[-1] - row['qty_roll_3']
        row['qty_volatility'] = np.std(last_vals[-6:])
        row['qty_lag1_x_sin'] = row['qty_lag_1'] * row['month_sin']
        row['qty_lag1_x_cos'] = row['qty_lag_1'] * row['month_cos']
        row['total_qty'] = last_vals[-1]
        row['total_revenue'] = 0

        X_future = pd.DataFrame([row])[FEATURE_COLS]
        pred = max(model.predict(X_future)[0], 0)
        preds.append(pred)

        row['total_revenue'] = pred
        extended = pd.concat([extended, pd.DataFrame([row])], ignore_index=True)

    return np.array(preds)


def _predict_customer_breakdown(df, total_preds):
    """Predict 2026 delivery qty for top 15 customers."""
    # Calculate recent share for each customer (2024-2025)
    recent = df[df['year'] >= 2024]
    cust_share = (recent.groupby('Customer name')['Delivery qty.']
                  .sum()
                  .nlargest(15))
    total = cust_share.sum()
    cust_share = cust_share / total

    total_2026 = sum(total_preds)
    rows = []
    for cust, share in cust_share.items():
        rows.append({
            'Customer': cust,
            'Share (%)': round(share * 100, 2),
            'Predicted 2026 Qty': round(share * total_2026),
        })
    return pd.DataFrame(rows)


def _predict_material_breakdown(df, total_preds):
    """Predict 2026 delivery qty for top 15 materials."""
    recent = df[df['year'] >= 2024]
    mat_share = (recent.groupby('Material name')['Delivery qty.']
                 .sum()
                 .nlargest(15))
    total = mat_share.sum()
    mat_share = mat_share / total

    total_2026 = sum(total_preds)
    rows = []
    for mat, share in mat_share.items():
        rows.append({
            'Material': mat,
            'Share (%)': round(share * 100, 2),
            'Predicted 2026 Qty': round(share * total_2026),
        })
    return pd.DataFrame(rows)


def _predict_customer_monthly(df, predictions_qty, predictions_rev):
    """Predict per-customer monthly Qty and Revenue for 2026.

    Uses historical monthly share patterns from 2024-2025 to distribute
    the overall monthly predictions across top 15 customers + Others.
    Shares are normalized so column sums = overall monthly predictions.
    """
    recent = df[df['year'] >= 2024]
    top_customers = (recent.groupby('Customer name')['Delivery qty.']
                     .sum().nlargest(15).index.tolist())

    # Per-customer per-month aggregates (top 15 only)
    top_data = recent[recent['Customer name'].isin(top_customers)]
    cust_month = top_data.groupby(['Customer name', 'month_num']).agg(
        qty=('Delivery qty.', 'sum'),
        rev=('Value of supply(KRW)', 'sum'),
    ).reset_index()

    # All-customer monthly totals
    all_month = recent.groupby('month_num').agg(
        total_qty=('Delivery qty.', 'sum'),
        total_rev=('Value of supply(KRW)', 'sum'),
    ).reset_index()

    months = list(range(1, 13))
    month_labels = [f'2026-{m:02d}' for m in months]

    qty_rows = []
    rev_rows = []
    # Track per-month sums for Others row
    month_qty_allocated = {m: 0.0 for m in months}
    month_rev_allocated = {m: 0.0 for m in months}

    for cust in top_customers:
        qty_row = {'Customer': cust}
        rev_row = {'Customer': cust}
        annual_qty = 0.0
        annual_rev = 0.0

        for m in months:
            # Get this customer's share of ALL customers for month m
            cm = cust_month[(cust_month['Customer name'] == cust) &
                            (cust_month['month_num'] == m)]
            am = all_month[all_month['month_num'] == m]

            if len(cm) > 0 and len(am) > 0:
                q_share = cm['qty'].values[0] / max(am['total_qty'].values[0], 1)
                r_share = cm['rev'].values[0] / max(am['total_rev'].values[0], 1)
            else:
                q_share = 0.0
                r_share = 0.0

            pred_q = predictions_qty[m - 1] * q_share
            pred_r = predictions_rev[m - 1] * r_share
            qty_row[month_labels[m - 1]] = round(pred_q)
            rev_row[month_labels[m - 1]] = round(pred_r)
            annual_qty += pred_q
            annual_rev += pred_r
            month_qty_allocated[m] += pred_q
            month_rev_allocated[m] += pred_r

        qty_row['Total 2026'] = round(annual_qty)
        rev_row['Total 2026'] = round(annual_rev)
        qty_rows.append(qty_row)
        rev_rows.append(rev_row)

    # Add "Others" row = overall prediction - top 15 allocated
    others_qty = {'Customer': 'Others'}
    others_rev = {'Customer': 'Others'}
    others_annual_qty = 0.0
    others_annual_rev = 0.0
    for m in months:
        oq = predictions_qty[m - 1] - month_qty_allocated[m]
        orv = predictions_rev[m - 1] - month_rev_allocated[m]
        others_qty[month_labels[m - 1]] = round(max(oq, 0))
        others_rev[month_labels[m - 1]] = round(max(orv, 0))
        others_annual_qty += max(oq, 0)
        others_annual_rev += max(orv, 0)
    others_qty['Total 2026'] = round(others_annual_qty)
    others_rev['Total 2026'] = round(others_annual_rev)
    qty_rows.append(others_qty)
    rev_rows.append(others_rev)

    return pd.DataFrame(qty_rows), pd.DataFrame(rev_rows)


def _predict_customer_material_monthly(df, predictions_qty):
    """Predict per-customer per-material monthly Qty for 2026.

    For each top customer, identifies their top materials and distributes
    the overall monthly predictions using historical (customer, material, month)
    share patterns from 2024-2025.

    Returns DataFrame with columns:
        Customer | Material | 2026-01 | ... | 2026-12 | Total 2026
    """
    recent = df[df['year'] >= 2024]
    top_customers = (recent.groupby('Customer name')['Delivery qty.']
                     .sum().nlargest(15).index.tolist())

    # All-customer monthly totals (denominator for share calc)
    all_month = recent.groupby('month_num').agg(
        total_qty=('Delivery qty.', 'sum'),
    ).reset_index()

    months = list(range(1, 13))
    month_labels = [f'2026-{m:02d}' for m in months]
    TOP_MATERIALS_PER_CUSTOMER = 10

    rows = []

    for cust in top_customers:
        cust_data = recent[recent['Customer name'] == cust]

        # Top materials for this customer
        top_materials = (cust_data.groupby('Material name')['Delivery qty.']
                         .sum().nlargest(TOP_MATERIALS_PER_CUSTOMER).index.tolist())

        for mat in top_materials:
            row = {'Customer': cust, 'Material': mat}
            annual_qty = 0.0

            for m in months:
                # Get (customer, material, month) qty from history
                cm_data = cust_data[
                    (cust_data['Material name'] == mat) &
                    (cust_data['month_num'] == m)
                ]['Delivery qty.'].sum()

                # Get total all-customer qty for this month
                am = all_month[all_month['month_num'] == m]
                total_m = am['total_qty'].values[0] if len(am) > 0 else 1

                # Share = this (cust, mat, month) / all customers total for month
                share = cm_data / max(total_m, 1)

                pred_q = predictions_qty[m - 1] * share
                row[month_labels[m - 1]] = round(pred_q)
                annual_qty += pred_q

            row['Total 2026'] = round(annual_qty)
            # Only include rows with non-zero prediction
            if annual_qty > 0:
                rows.append(row)

    result_df = pd.DataFrame(rows)
    # Sort by Total 2026 descending
    if len(result_df) > 0:
        result_df = result_df.sort_values('Total 2026', ascending=False).reset_index(drop=True)

    return result_df


def _chart_predictions(monthly, future_dates, preds_qty, preds_rev, model_name):
    """Chart historical + 2026 prediction."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # --- Delivery Qty ---
    ax1.plot(monthly['date'], monthly['total_qty'],
             color=PALETTE[0], linewidth=2, label='Historical')
    ax1.plot(future_dates, preds_qty,
             color=PALETTE[3], linewidth=2.5, marker='D', markersize=6,
             label=f'2026 Forecast ({model_name})')

    # Confidence band (±15%)
    lower = preds_qty * 0.85
    upper = preds_qty * 1.15
    ax1.fill_between(future_dates, lower, upper,
                     alpha=0.15, color=PALETTE[3], label='±15% Confidence')

    ax1.axvline(x=pd.Timestamp('2026-01-01'), color='#ffffff',
                linestyle=':', alpha=0.4)
    ax1.set_title('Delivery Quantity: Historical + 2026 Forecast', fontweight='bold')
    ax1.set_ylabel('Delivery Qty')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x / 1e6:.1f}M'))
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # --- Revenue ---
    ax2.plot(monthly['date'], monthly['total_revenue'],
             color=PALETTE[1], linewidth=2, label='Historical')
    ax2.plot(future_dates, preds_rev,
             color=PALETTE[4], linewidth=2.5, marker='D', markersize=6,
             label='2026 Revenue Forecast')
    lower_r = preds_rev * 0.85
    upper_r = preds_rev * 1.15
    ax2.fill_between(future_dates, lower_r, upper_r,
                     alpha=0.15, color=PALETTE[4], label='±15% Confidence')
    ax2.axvline(x=pd.Timestamp('2026-01-01'), color='#ffffff',
                linestyle=':', alpha=0.4)
    ax2.set_title('Revenue (KRW): Historical + 2026 Forecast', fontweight='bold')
    ax2.set_ylabel('Revenue (KRW)')
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x / 1e9:.1f}B'))
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '08_2026_forecast.png'), dpi=150)
    plt.close(fig)
    print("  → 08_2026_forecast.png")


def _chart_customer_forecast(df, total_preds):
    """Top customer predicted 2026 breakdown."""
    recent = df[df['year'] >= 2024]
    cust_share = (recent.groupby('Customer name')['Delivery qty.']
                  .sum()
                  .nlargest(10))
    total = cust_share.sum()
    cust_share = cust_share / total
    total_2026 = sum(total_preds)
    pred_vals = cust_share * total_2026

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(pred_vals)), pred_vals.sort_values().values,
                   color=PALETTE[:len(pred_vals)][::-1],
                   edgecolor='white', linewidth=0.3)
    ax.set_yticks(range(len(pred_vals)))
    ax.set_yticklabels([n[:40] for n in pred_vals.sort_values().index], fontsize=9)
    ax.set_title('Predicted 2026 Delivery Qty by Top 10 Customers', fontweight='bold')
    ax.set_xlabel('Predicted Delivery Qty')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x / 1e6:.1f}M'))
    for bar, val in zip(bars, pred_vals.sort_values().values):
        ax.text(val, bar.get_y() + bar.get_height() / 2,
                f'  {val / 1e6:.1f}M', va='center', fontsize=9, color='#e0e0e0')
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '09_customer_forecast_2026.png'), dpi=150)
    plt.close(fig)
    print("  → 09_customer_forecast_2026.png")


def _chart_customer_material_heatmap(cust_mat_df):
    """Heatmap: top customers × top materials predicted 2026 qty."""
    if len(cust_mat_df) == 0:
        print("  [!] No customer-material data for heatmap")
        return

    # Aggregate to customer × material totals
    pivot = cust_mat_df.pivot_table(
        index='Customer', columns='Material',
        values='Total 2026', aggfunc='sum', fill_value=0
    )

    # Keep top 10 customers and top 10 materials by total
    top_custs = pivot.sum(axis=1).nlargest(10).index
    top_mats = pivot.sum(axis=0).nlargest(10).index
    pivot = pivot.loc[
        pivot.index.isin(top_custs),
        pivot.columns.isin(top_mats)
    ]

    # Shorten labels
    pivot.index = [n[:35] for n in pivot.index]
    pivot.columns = [n[:25] for n in pivot.columns]

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(
        pivot / 1e6, annot=True, fmt='.1f', cmap='magma_r',
        linewidths=0.5, linecolor='#2a2a4a',
        cbar_kws={'label': 'Predicted Qty (M)'},
        ax=ax
    )
    ax.set_title('2026 Predicted Delivery Qty: Customer × Material (Millions)',
                 fontweight='bold', pad=15)
    ax.set_xlabel('Material')
    ax.set_ylabel('Customer')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, '10_customer_material_heatmap.png'), dpi=150)
    plt.close(fig)
    print("  → 10_customer_material_heatmap.png")


# ============================================================
# MAIN
# ============================================================
def main():
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("+------------------------------------------------------+")
    print("|  Seoul Semiconductor -- Order Prediction Pipeline     |")
    print("|  Data: 2021-2025 -> Forecast: 2026                   |")
    print("+------------------------------------------------------+")

    setup_dirs()
    df = load_data()
    run_eda(df)
    monthly = build_monthly_features(df)
    results, best_model_name, monthly = train_models(monthly)
    pred_df = predict_2026(results, best_model_name, monthly, df)

    print("\n" + "=" * 60)
    print("DONE! All outputs saved to output/")
    print("=" * 60)
    print("\n2026 Predictions Summary:")
    print(pred_df.to_string(index=False))
    print(f"\nCharts:      {CHARTS_DIR}/")
    print(f"Predictions: {os.path.join(OUTPUT_DIR, 'predictions_2026.xlsx')}")
    print(f"Report:      {os.path.join(OUTPUT_DIR, 'model_report.txt')}")


if __name__ == '__main__':
    main()

