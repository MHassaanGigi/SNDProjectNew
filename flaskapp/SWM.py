import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but .* was fitted with feature names"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="DataFrame.fillna with 'method' is deprecated.*"
)

import pandas as pd
import numpy as np
from db_connect import run_query
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import category_encoders as ce
from sklearn.metrics import r2_score

# ======================================
# CONFIG
# ======================================
LAGS = [1, 2, 3]
ROLLING_WINDOWS = [3, 7, 14]
FEATURES = [
    'MONTH', 'QUARTER',
    'lag_1', 'lag_2', 'lag_3',
    'ROLLING_MEAN_3', 'ROLLING_STD_3',
    'ROLLING_MEAN_7', 'ROLLING_MEAN_14',
    'growth_1', 'growth_3',
    'TOWN_ENC', 'SKU_CODE_ENC'
]
MODELS = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.1, subsample=1.0, random_state=42)
}

# ======================================
# 1. Data Extraction
# ======================================
def get_sku_data(division='CANDYLAND', sku_pattern='%24 BAGS X 24 BOXES%', territory='KARACHI'):
    query = f"""
        SELECT 
            YEAR,
            MONTH,
            TOWN,
            SKU_CODE,
            SUM(CAST(NET_SALES_UNITS AS INT)) AS TARGET
        FROM FINAL_QUERY
        WHERE DIVISION = '{division}'
          AND SKU_LDESC LIKE '{sku_pattern}'
          AND TERRITORY = '{territory}'
          AND NET_SALES_UNITS > 0
          AND (YEAR < 2025 OR (YEAR = 2025 AND MONTH <= 4))
        GROUP BY YEAR, MONTH, TOWN, SKU_CODE
        ORDER BY YEAR, MONTH, TOWN, SKU_CODE
    """
    return run_query(query, as_dataframe=True)

def get_last_year_data(division='CANDYLAND', sku_pattern='%24 BAGS X 24 BOXES%', territory='KARACHI'):
        query = f"""
            SELECT 
                YEAR,
                MONTH,
                SUM(CAST(NET_SALES_UNITS AS INT)) AS TARGET
            FROM FINAL_QUERY
            WHERE DIVISION = '{division}'
            AND SKU_LDESC LIKE '{sku_pattern}'
            AND TERRITORY = '{territory}'
            AND NET_SALES_UNITS > 0
            AND YEAR = 2024  -- last year
            GROUP BY YEAR, MONTH
            ORDER BY YEAR, MONTH
        """
        df = run_query(query, as_dataframe=True)
        df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str))
        df['MONTH'] = df['DATE'].dt.strftime('%Y-%b')
        return df

# ======================================
# 2. Feature Engineering
# ======================================
def prepare_data(df=None):
    if df is None:
        df = get_sku_data()

    df.columns = df.columns.str.strip().str.upper()
    df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str))
    df['QUARTER'] = df['DATE'].dt.quarter
    df = df.sort_values(['TOWN', 'DATE'])
    df['TARGET_LOG'] = np.log1p(df['TARGET'])

    # Lags
    for lag in LAGS:
        df[f'lag_{lag}'] = df.groupby(['SKU_CODE', 'TOWN'])['TARGET_LOG'].shift(lag)

    # Rolling features
    df['ROLLING_MEAN_3'] = df.groupby(['SKU_CODE','TOWN'])['TARGET_LOG'].transform(lambda x: x.shift(1).rolling(3).mean())
    df['ROLLING_STD_3']  = df.groupby(['SKU_CODE','TOWN'])['TARGET_LOG'].transform(lambda x: x.shift(1).rolling(3).std())
    df['ROLLING_MEAN_7'] = df.groupby(['SKU_CODE','TOWN'])['TARGET_LOG'].transform(lambda x: x.shift(1).rolling(7).mean())
    df['ROLLING_MEAN_14'] = df.groupby(['SKU_CODE','TOWN'])['TARGET_LOG'].transform(lambda x: x.shift(1).rolling(14).mean())

    # Growth features
    df['growth_1'] = np.divide(df['TARGET_LOG'], df['lag_1'], out=np.zeros_like(df['TARGET_LOG']), where=df['lag_1']!=0) - 1
    df['growth_3'] = np.divide(df['TARGET_LOG'], df['lag_3'], out=np.zeros_like(df['TARGET_LOG']), where=df['lag_3']!=0) - 1

    # Town frequency encoding
    df['TOWN_ENC'] = df['TOWN'].map(df['TOWN'].value_counts())

    return df

# ======================================
# 3. Forecast Function (Merged)
# ======================================
def run_full_forecast(months_ahead=8, debug=False):
    df = prepare_data()

    # Drop rows missing essential lags/stats
    essential_cols = ['lag_1','lag_2','lag_3','ROLLING_MEAN_3','ROLLING_STD_3','TARGET_LOG']
    df.dropna(subset=essential_cols, inplace=True)

    # Fill minor NaNs
    df['growth_1'] = df['growth_1'].fillna(0)
    df['growth_3'] = df['growth_3'].fillna(0)
    df['ROLLING_MEAN_14'] = df['ROLLING_MEAN_14'].fillna(method='bfill')

    # Train/test split
    train_end = pd.to_datetime("2025-02")
    test_start = pd.to_datetime("2025-03")
    test_end   = pd.to_datetime("2025-04")
    train_df = df[df['DATE'] <= train_end]
    test_df  = df[(df['DATE'] >= test_start) & (df['DATE'] <= test_end)]

    # Encode SKUs
    sku_encoder = ce.TargetEncoder(cols=['SKU_CODE'])
    sku_encoder.fit(train_df['SKU_CODE'], train_df['TARGET_LOG'])
    train_df['SKU_CODE_ENC'] = sku_encoder.transform(train_df['SKU_CODE'])
    test_df['SKU_CODE_ENC']  = sku_encoder.transform(test_df['SKU_CODE'])

    # Train models
    X_train, y_train = train_df[FEATURES], train_df['TARGET_LOG']
    X_test, y_test   = test_df[FEATURES], test_df['TARGET_LOG']

    trained_models = {}
    results = {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        results[name] = {
            'train': model.predict(X_train),
            'test': model.predict(X_test)
        }
        if debug:
            print(f"{name} R² Train: {r2_score(y_train, results[name]['train']):.4f} | Test: {r2_score(y_test, results[name]['test']):.4f}")

    # Build March–April comparison df
    def expm1_series(s): return np.expm1(s)
    comparison_df = pd.DataFrame({
        'DATE': test_df['DATE'],
        'Actual': expm1_series(y_test),
        'RandomForest': expm1_series(results['RandomForest']['test']),
        'XGBoost': expm1_series(results['XGBoost']['test'])
    }).groupby('DATE').sum(numeric_only=True).reset_index()
    
    comparison_df['month_date'] = comparison_df['DATE']  # ✅ sortable date
    comparison_df['MONTH'] = comparison_df['month_date'].dt.strftime('%Y-%b')
    comparison_df = comparison_df.sort_values('month_date')  # ✅ sorted here

    # Forward forecast (including March–April preds, but without actuals)
    last_date = df['DATE'].max()
    forward_preds = []
    sku_preds = []
    town_preds = []

    # Start with March–April predictions
    for _, row in test_df.iterrows():
        month_label = row['DATE'].strftime('%Y-%b')
        rf_pred = np.expm1(trained_models['RandomForest'].predict([row[FEATURES]])[0])
        xgb_pred = np.expm1(trained_models['XGBoost'].predict([row[FEATURES]])[0])
        forward_preds.append({
            'MONTH': month_label,
            'month_date': row['DATE'],  # ✅ keep actual datetime for sorting
            'PREDICTED_RF': round(rf_pred),
            'PREDICTED_XGB': round(xgb_pred)
        })

    # Generate future months
    skus = df['SKU_CODE'].unique()
    towns = df['TOWN'].unique()
    # Generate future months with rolling predictions
    # Prepare a tracker for the latest lag values per SKU/TOWN
    latest_values = {}
    for sku in skus:
        for town in towns:
            last_row = df[(df['SKU_CODE'] == sku) & (df['TOWN'] == town)].iloc[-1]
            latest_values[(sku, town)] = {
                'lag_1': last_row['TARGET_LOG'],
                'lag_2': last_row['lag_1'],
                'lag_3': last_row['lag_2'],
                'ROLLING_MEAN_3': last_row['ROLLING_MEAN_3'],
                'ROLLING_STD_3': last_row['ROLLING_STD_3'],
                'ROLLING_MEAN_7': last_row['ROLLING_MEAN_7'],
                'ROLLING_MEAN_14': last_row['ROLLING_MEAN_14'],
                'growth_1': last_row['growth_1'],
                'growth_3': last_row['growth_3'],
                'TOWN_ENC': last_row['TOWN_ENC'],
                'SKU_CODE_ENC': sku_encoder.transform(pd.DataFrame({'SKU_CODE': [sku]})).iloc[0, 0]
            }

    # Generate future months
    for m in range(1, months_ahead+1):
        next_month = last_date + pd.DateOffset(months=m)

        for (sku, town), lv in latest_values.items():
            row_dict = {
                'MONTH': next_month.month,
                'QUARTER': next_month.quarter,
                'TOWN_ENC': lv['TOWN_ENC'],
                'SKU_CODE_ENC': lv['SKU_CODE_ENC'],
                'lag_1': lv['lag_1'],
                'lag_2': lv['lag_2'],
                'lag_3': lv['lag_3'],
                'ROLLING_MEAN_3': lv['ROLLING_MEAN_3'],
                'ROLLING_STD_3': lv['ROLLING_STD_3'],
                'ROLLING_MEAN_7': lv['ROLLING_MEAN_7'],
                'ROLLING_MEAN_14': lv['ROLLING_MEAN_14'],
                'growth_1': lv['growth_1'],
                'growth_3': lv['growth_3']
            }
            X_pred = pd.DataFrame([row_dict])[FEATURES]
            X_pred = X_pred.fillna(method='bfill').fillna(method='ffill').fillna(0)

            rf_pred_log = trained_models['RandomForest'].predict(X_pred)[0]
            xgb_pred_log = trained_models['XGBoost'].predict(X_pred)[0]

            rf_pred = np.expm1(rf_pred_log)
            xgb_pred = np.expm1(xgb_pred_log)

            forward_preds.append({
                'MONTH': next_month.strftime('%Y-%b'),
                'month_date': next_month,
                'PREDICTED_RF': round(rf_pred),
                'PREDICTED_XGB': round(xgb_pred)
            })

            sku_preds.append({
                'SKU': sku,
                'MONTH': next_month.strftime('%Y-%b'),
                'month_date': next_month,
                'PREDICTED_RF': round(rf_pred),
                'PREDICTED_XGB': round(xgb_pred)
            })
            
            town_preds.append({
                'TOWN': town,
                'MONTH': next_month.strftime('%Y-%b'),
                'month_date': next_month,
                'PREDICTED_RF': round(rf_pred),
                'PREDICTED_XGB': round(xgb_pred)
            })

            # ---- UPDATE latest_values for next month's prediction ----
            # Shift lags
            lv['lag_3'] = lv['lag_2']
            lv['lag_2'] = lv['lag_1']
            avg_pred_log = (rf_pred_log + xgb_pred_log) / 2
            lv['lag_1'] = avg_pred_log  # next month uses mean of RF/XGB predictions
            row_dict['lag_1'] = avg_pred_log  # also update for current prediction
            # Update rolling means (simple append-shift window logic)
            # but here we approximate with weighted average updates
            lv['ROLLING_MEAN_3'] = ((lv['ROLLING_MEAN_3'] * 2) + rf_pred_log) / 3
            lv['ROLLING_MEAN_7'] = ((lv['ROLLING_MEAN_7'] * 6) + rf_pred_log) / 7
            lv['ROLLING_MEAN_14'] = ((lv['ROLLING_MEAN_14'] * 13) + rf_pred_log) / 14
            lv['ROLLING_STD_3'] = 0  # Optional: could maintain if needed

            # Update growth rates
            lv['growth_1'] = (rf_pred_log / lv['lag_2']) - 1 if lv['lag_2'] != 0 else 0
            lv['growth_3'] = (rf_pred_log / lv['lag_3']) - 1 if lv['lag_3'] != 0 else 0

    forecast_df = pd.DataFrame(forward_preds)
    forecast_df = forecast_df.groupby(['MONTH', 'month_date'])[['PREDICTED_RF', 'PREDICTED_XGB']].sum().reset_index()
    forecast_df = forecast_df.sort_values('month_date')  # ✅ sorted by real date
    
    sku_df = pd.DataFrame(sku_preds)
    sku_df = sku_df.groupby(['SKU','MONTH', 'month_date'])[['PREDICTED_RF', 'PREDICTED_XGB']].sum().reset_index()
    sku_df = sku_df.sort_values('month_date')  # ✅ sorted by real date
    
    town_df = pd.DataFrame(town_preds)
    town_df = town_df.groupby(['TOWN','MONTH', 'month_date'])[['PREDICTED_RF', 'PREDICTED_XGB']].sum().reset_index()
    town_df = town_df.sort_values('month_date')  # ✅ sorted by real date
    
    # ===== DEBUG PRINT =====
    print("\n=== FINAL FORECAST DEBUG ===")
    print(f"Total forecast rows: {len(forecast_df)}")
    print(forecast_df.tail(10))  # Last 10 rows for quick inspection
    print("\nMin/Max Predicted_RF:", forecast_df['PREDICTED_RF'].min(), forecast_df['PREDICTED_RF'].max())
    print("Min/Max Predicted_XGB:", forecast_df['PREDICTED_XGB'].min(), forecast_df['PREDICTED_XGB'].max())
    print("============================\n")
    print("Comparison DF columns:", comparison_df.columns.tolist())
    print("Forecast DF columns:", forecast_df.columns.tolist())

    return comparison_df, forecast_df,sku_df,town_df
