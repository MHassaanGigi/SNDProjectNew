# models/sku_forecasting.py

import pandas as pd
import numpy as np
from db_connect import run_query
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import category_encoders as ce


#Basic Query
def get_sku_data():
    query = """
   SELECT 
    YEAR,
	MONTH,
	TOWN,
    SKU_CODE,
	sum(cast(NET_SALES_UNITS AS INT)) AS TARGET
FROM
	FINAL_QUERY
WHERE
	DIVISION = 'CANDYLAND'
	AND SKU_LDESC LIKE '%24 BAGS X 24 BOXES%'
    AND TERRITORY = 'KARACHI'
    AND NET_SALES_UNITS > 0
GROUP BY
    YEAR,
	MONTH,
	TOWN,
    SKU_CODE
ORDER BY
    YEAR,MONTH,TOWN,SKU_CODE
     """
    return run_query(query, as_dataframe=True)

df = get_sku_data()

def prepare_data():
    df = get_sku_data()
    print(f"Initial data shape: {df.shape}")
    print(f"Unique SKUs: {df['SKU_CODE'].nunique()}")
    print(f"Unique Towns: {df['TOWN'].nunique()}")
    print(f"Sample SKUs: {df['SKU_CODE'].unique()}")
    print(f"Sample Towns: {df['TOWN'].unique()}")
    df.columns = df.columns.str.strip().str.upper()

    df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str))

    df['QUARTER'] = df['DATE'].dt.quarter
    df = df.sort_values(['TOWN', 'DATE'])

    #lower = df['TARGET'].quantile(0.03)
    #upper = df['TARGET'].quantile(0.97)
    df['TARGET_winsor'] = df['TARGET'] #.clip(lower, upper)
    df['TARGET_LOG'] = np.log1p(df['TARGET_winsor'])

    for lag in [1, 2, 3]:
        df[f'lag_{lag}'] = df.groupby(['SKU_CODE', 'TOWN'])['TARGET_LOG'].shift(lag)

    df['ROLLING_MEAN_3'] = df.groupby(['SKU_CODE','TOWN'])['TARGET_LOG'].shift(1).rolling(window=3).mean()
    df['ROLLING_STD_3'] = df.groupby(['SKU_CODE','TOWN'])['TARGET_LOG'].shift(1).rolling(window=3).std()
    df['ROLLING_MEAN_7'] = df.groupby(['SKU_CODE', 'TOWN'])['TARGET_LOG'].transform(lambda x: x.shift(1).rolling(window=7).mean())
    df['ROLLING_MEAN_14'] = df.groupby(['SKU_CODE', 'TOWN'])['TARGET_LOG'].transform(lambda x: x.shift(1).rolling(window=14).mean())

    df['START_OF_YEAR'] = (df['DATE'].dt.month == 1).astype(int)
    df['END_OF_YEAR'] = (df['DATE'].dt.month == 12).astype(int)

    df['growth_1'] = df['TARGET_LOG'] / df['lag_1'] - 1
    df['growth_3'] = df['TARGET_LOG'] / df['lag_3'] - 1
    df['TOWN_FREQ'] = df['TOWN'].map(df['TOWN'].value_counts())
    return df

def train_models(X_train, y_train, X_test, y_test):
    X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
    X_test = X_test.replace([np.inf, -np.inf], np.nan).dropna()

    y_train = y_train.loc[X_train.index]
    y_test = y_test.loc[X_test.index]

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.1, subsample=1.0)
    }

    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        predictions[name] = {
            "train": pred_train,
            "test": pred_test,
            "model": model
        }
    from sklearn.metrics import r2_score
    print("R² Scores:")
    for name, preds in predictions.items():
        r2_train = r2_score(y_train, preds['train'])
        r2_test = r2_score(y_test, preds['test'])
        print(f"{name} - Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}")

    return predictions


def run_forecasting():
    df = prepare_data()
    
    train_end = pd.to_datetime("2025-02")
    test_start = pd.to_datetime("2025-03")
    test_end = pd.to_datetime("2025-04")

    features = [
        'MONTH', 'QUARTER',
        'lag_1', 'lag_2', 'lag_3', 'ROLLING_MEAN_3',
        'ROLLING_STD_3', 'ROLLING_MEAN_7', 'ROLLING_MEAN_14',
        'growth_1', 'growth_3', 'TOWN_ENC', 'SKU_CODE_ENC'
    ]

    # Drop only essential missing values
    essential_cols = ['lag_1', 'lag_2', 'lag_3', 'ROLLING_MEAN_3', 'ROLLING_STD_3', 'TARGET_LOG']
    df.dropna(subset=essential_cols, inplace=True)

    # Optionally fill less important NaNs
    df['growth_1'] = df['growth_1'].fillna(0)
    df['growth_3'] = df['growth_3'].fillna(0)
    df['ROLLING_MEAN_14'] = df['ROLLING_MEAN_14'].fillna(method='bfill')

    train_df = df[df['DATE'] <= train_end]
    test_df = df[(df['DATE'] >= test_start) & (df['DATE'] <= test_end)]

    # ✅ Target encoding — fit ONLY on training data
    sku_encoder = ce.TargetEncoder(cols=['SKU_CODE'])
    sku_encoder.fit(train_df['SKU_CODE'], train_df['TARGET_LOG'])

    train_df['SKU_CODE_ENC'] = sku_encoder.transform(train_df['SKU_CODE'])
    test_df['SKU_CODE_ENC'] = sku_encoder.transform(test_df['SKU_CODE'])

    # You can keep TOWN_FREQ as-is, no leakage
    train_df.rename(columns={'TOWN_FREQ': 'TOWN_ENC'}, inplace=True)
    test_df.rename(columns={'TOWN_FREQ': 'TOWN_ENC'}, inplace=True)

    X_train, y_train = train_df[features], train_df['TARGET_LOG']
    X_test, y_test = test_df[features], test_df['TARGET_LOG']

    results = train_models(X_train, y_train, X_test, y_test)
    trained_models = {name: info['model'] for name, info in results.items()}

    # Get min length of all series for train
    train_min_len = min(
        len(train_df["DATE"]),
        len(train_df["TARGET_LOG"]),
        len(results["RandomForest"]["train"]),
        len(results["XGBoost"]["train"])
    )

    train_output = pd.DataFrame({
        "DATE": train_df["DATE"].values[:train_min_len],
        "ACTUAL": train_df["TARGET_LOG"].values[:train_min_len],
        "RF_PRED": results["RandomForest"]["train"][:train_min_len],
        "XGB_PRED": results["XGBoost"]["train"][:train_min_len]
    })

    # Same for test
    test_min_len = min(
        len(test_df["DATE"]),
        len(y_test),
        len(results["RandomForest"]["test"]),
        len(results["XGBoost"]["test"])
    )

    test_output = pd.DataFrame({
        "DATE": test_df["DATE"].values[:test_min_len],
        "ACTUAL": y_test.values[:test_min_len],
        "RF_PRED": results["RandomForest"]["test"][:test_min_len],
        "XGB_PRED": results["XGBoost"]["test"][:test_min_len]
    })

    comparison_df = pd.DataFrame({
        'DATE': pd.concat([
            train_df["DATE"].iloc[:train_min_len].reset_index(drop=True),
            test_df["DATE"].iloc[:test_min_len].reset_index(drop=True)
        ]),
        'Actual': pd.concat([
            pd.Series(np.expm1(y_train[:train_min_len])).reset_index(drop=True),
            pd.Series(np.expm1(y_test[:test_min_len])).reset_index(drop=True)
        ]),
        'RandomForest': pd.concat([
            pd.Series(np.expm1(results["RandomForest"]["train"][:train_min_len])).reset_index(drop=True),
            pd.Series(np.expm1(results["RandomForest"]["test"][:test_min_len])).reset_index(drop=True)
        ]),
        'XGBoost': pd.concat([
            pd.Series(np.expm1(results["XGBoost"]["train"][:train_min_len])).reset_index(drop=True),
            pd.Series(np.expm1(results["XGBoost"]["test"][:test_min_len])).reset_index(drop=True)
        ])
    })

    # Convert DATE to datetime and filter
    comparison_df['DATE'] = pd.to_datetime(comparison_df['DATE'])
    comparison_df = comparison_df[
        (comparison_df['DATE'] >= '2025-03') & (comparison_df['DATE'] <= '2025-04')
    ]

    # Set DATE as index for resampling
    comparison_df.set_index('DATE', inplace=True)

    monthly_df = comparison_df.resample('M').sum(numeric_only=True).reset_index()

    # Step 6: Add readable MONTH labels (e.g., "2025-Jan")
    monthly_df['MONTH'] = monthly_df['DATE'].dt.strftime('%Y-%b')  # e.g., "2025-Jan"

    # Step 7: Print descriptive summaries of model predictions
    print("\n=== DEBUG: Prediction Summary ===")
    
    print("RandomForest Predictions (train):")
    print(pd.Series(results["RandomForest"]["train"][:train_min_len]).describe())

    print("RandomForest Predictions (test):")
    print(pd.Series(results["RandomForest"]["test"][:test_min_len]).describe())

    print("\nXGBoost Predictions (train):")
    print(pd.Series(results["XGBoost"]["train"][:train_min_len]).describe())

    print("XGBoost Predictions (test):")
    print(pd.Series(results["XGBoost"]["test"][:test_min_len]).describe())

    print("\nAny NaNs in predictions?")
    print("RandomForest (train):", pd.Series(results["RandomForest"]["train"]).isna().sum())
    print("RandomForest (test):", pd.Series(results["RandomForest"]["test"]).isna().sum())
    print("XGBoost (train):", pd.Series(results["XGBoost"]["train"]).isna().sum())
    print("XGBoost (test):", pd.Series(results["XGBoost"]["test"]).isna().sum())

    print("\nFinal Comparison DF Preview (first 10 rows):")
    print(monthly_df.head(10))

    
    return monthly_df,trained_models

def generate_forward_forecast(train_models,months_ahead=6):
    # Step 1: Prepare historical data
    df = prepare_data()
    df = df.replace([np.inf, -np.inf], np.nan)
    # Fill NaNs by backward fill, then forward fill as fallback
    df = df.fillna(method='bfill').fillna(method='ffill')
    # Optionally, fill any remaining NaNs with zero or some constant if needed
    df = df.fillna(0)

    feature_cols = [
        'MONTH', 'QUARTER',
        'lag_1', 'lag_2', 'lag_3', 'ROLLING_MEAN_3',
        'ROLLING_STD_3', 'ROLLING_MEAN_7', 'ROLLING_MEAN_14',
        'growth_1', 'growth_3', 'TOWN_ENC', 'SKU_CODE_ENC'
    ]

    target_col = 'TARGET_LOG'

    models = train_models
    # Step 3: Start forward forecast
    last_date = pd.to_datetime("2025-04")
    forecast_rows = []

    df_forecast = df.copy()
    print("Unique SKUs before forecasting:", df['SKU_CODE'].nunique())
    print("Unique Towns before forecasting:", df['TOWN'].nunique())
    print("Total rows in df:", len(df))
    print("Unique SKUs in df_forecast (initial):", df_forecast['SKU_CODE'].nunique())
    print("Unique Towns in df_forecast (initial):", df_forecast['TOWN'].nunique())

    # Assume feature_cols includes MONTH_SIN and MONTH_COS, add them if missing:
    forecast_months = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months_ahead, freq='MS')

    forecast_rows = []
    df_forecast = df.copy()

    for next_date in forecast_months:
        month_num = next_date.month
        new_rows_batch = []

        for (sku, town), group in df_forecast.groupby(['SKU_CODE', 'TOWN']):
            group = group.sort_values('DATE')

            # Use last 14 records to calculate rolling stats if available, else fallback gracefully
            last_data = group.tail(14)

            # Lags on TARGET_LOG, fallback to NaN if insufficient data
            lag_1 = last_data.iloc[-1]['TARGET_LOG'] if len(last_data) >= 1 else np.nan
            lag_2 = last_data.iloc[-2]['TARGET_LOG'] if len(last_data) >= 2 else np.nan
            lag_3 = last_data.iloc[-3]['TARGET_LOG'] if len(last_data) >= 3 else np.nan

            rolling_mean_3 = last_data['TARGET_LOG'].tail(3).mean() if len(last_data) >= 3 else np.nan
            rolling_std_3 = last_data['TARGET_LOG'].tail(3).std() if len(last_data) >= 3 else np.nan
            rolling_mean_7 = last_data['TARGET_LOG'].mean()  # last 14 is enough
            rolling_mean_14 = last_data['TARGET_LOG'].mean()  # same as above; adjust window if needed

            growth_1 = (lag_1 / lag_2 - 1) if pd.notna(lag_2) and lag_2 != 0 else 0
            growth_3 = (lag_1 / lag_3 - 1) if pd.notna(lag_3) and lag_3 != 0 else 0

            sku_enc = int(group.iloc[-1]['SKU_CODE_ENC'])
            town_enc = int(group.iloc[-1]['TOWN_ENC'])

            new_row = {
                'DATE': next_date,
                'SKU_CODE': sku,
                'TOWN': town,
                'MONTH': month_num,
                'QUARTER': next_date.quarter,
                'lag_1': lag_1,
                'lag_2': lag_2,
                'lag_3': lag_3,
                'ROLLING_MEAN_3': rolling_mean_3,
                'ROLLING_STD_3': rolling_std_3,
                'ROLLING_MEAN_7': rolling_mean_7,
                'ROLLING_MEAN_14': rolling_mean_14,
                'growth_1': growth_1,
                'growth_3': growth_3,
                'SKU_CODE_ENC': sku_enc,
                'TOWN_ENC': town_enc,
            }

            X_pred = pd.DataFrame([new_row])[feature_cols]
            X_pred = X_pred.apply(pd.to_numeric, errors='raise')

            preds = {name: model.predict(X_pred)[0] for name, model in models.items()}

            for model_name, pred_log in preds.items():
                forecast_rows.append({
                    'DATE': next_date,
                    'SKU_CODE': sku,
                    'TOWN': town,
                    'MODEL': model_name,
                    'PRED_TARGET': np.expm1(pred_log)
                })

            avg_pred_log = np.mean(list(preds.values()))
            new_row['TARGET_LOG'] = avg_pred_log  # update with forecasted log value

            new_rows_batch.append(new_row)

        # Append all forecast rows of this month to df_forecast once
        df_forecast = pd.concat([df_forecast, pd.DataFrame(new_rows_batch)], ignore_index=True)

    # After loop ends, convert forecast_rows to DataFrame outside the loop as before
    forecast_df = pd.DataFrame(forecast_rows)
    print("Unique SKUs in forecast:", forecast_df['SKU_CODE'].nunique())
    print("Unique Towns in forecast:", forecast_df['TOWN'].nunique())
    print("Total rows in forecast_df:", len(forecast_df))
    agg_df = forecast_df.groupby(['DATE', 'MODEL'], as_index=False)['PRED_TARGET'].sum()

    # Pivot so RF and XGB are in separate columns
    agg_df = agg_df.pivot(index='DATE', columns='MODEL', values='PRED_TARGET').reset_index()
    agg_df = agg_df.rename(columns={'RandomForest': 'PREDICTED_RF', 'XGBoost': 'PREDICTED_XGB'})
        # DEBUG: Print monthly aggregated predictions per model
    print("\n=== Forward Forecast Monthly Predictions Summary ===")

    # Show first few rows of aggregated DataFrame
    print(agg_df.head(10))

    # Also print descriptive stats per model
    for model_col in ['PREDICTED_RF', 'PREDICTED_XGB']:
        if model_col in agg_df.columns:
            print(f"\nDescriptive stats for {model_col}:")
            print(agg_df[model_col].describe())

    # Optional: print total predicted sum per model over forecast horizon
    total_rf = agg_df['PREDICTED_RF'].sum() if 'PREDICTED_RF' in agg_df else None
    total_xgb = agg_df['PREDICTED_XGB'].sum() if 'PREDICTED_XGB' in agg_df else None

    print(f"\nTotal predicted sum over {len(agg_df)} months:")
    print(f"  RandomForest: {total_rf}")
    print(f"  XGBoost: {total_xgb}")

    agg_df['DATE'] = pd.to_datetime(agg_df['DATE'])  # ensure datetime type
    agg_df.set_index('DATE', inplace=True)

    # Resample by month and sum predictions across days (if any daily granularity)
    monthly_df = agg_df.resample('M').sum(numeric_only=True).reset_index()

    # Add readable month labels like "2025-Jan"
    monthly_df['MONTH'] = monthly_df['DATE'].dt.strftime('%Y-%b')

    # Optional: print summary
    print("\n=== Monthly Aggregated Forecast ===")
    print(monthly_df.head())

    return monthly_df


