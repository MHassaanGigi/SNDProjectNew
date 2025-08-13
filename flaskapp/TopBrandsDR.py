import pandas as pd
import numpy as np
from db_connect import run_query
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

def get_sales_forecast_data(var_division=None):
    # === Load Data with Dynamic Filters ===
    base_query = """
    SELECT REGION, DIVISION, BRAND, YEAR, MONTH, SUM(NET_SALE_AMOUNT) AS NET_SALE_AMOUNT
    FROM FINAL_QUERY
    """
    
    conditions = []
    if var_division:
        conditions.append(f"DIVISION = '{var_division}'")
    
    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)
    
    base_query += """
    GROUP BY REGION, DIVISION, BRAND, YEAR, MONTH
    ORDER BY REGION, DIVISION, BRAND, YEAR, MONTH
    """
    
    raw_df = pd.DataFrame(run_query(base_query))
    
    # === Top 3 Brands ===
    top_brands = raw_df.groupby("BRAND")['NET_SALE_AMOUNT'].sum().reset_index()
    top_3_brands = top_brands.sort_values("NET_SALE_AMOUNT", ascending=False).head(3)["BRAND"].tolist()
    raw_df = raw_df[raw_df["BRAND"].isin(top_3_brands)]
    
    # === Feature Preparation ===
    def prepare_features(df):
        df["DATE"] = pd.to_datetime(df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str).str.zfill(2) + "-01")
        df = df.sort_values(["REGION", "BRAND", "DATE"])
        df["NET_SALE_AMOUNT"] = pd.to_numeric(df["NET_SALE_AMOUNT"], errors="coerce").fillna(0)
        df.loc[df["NET_SALE_AMOUNT"] < 0, "NET_SALE_AMOUNT"] = 0
    
        all_dates = pd.date_range("2023-01-01", "2025-12-01", freq="MS")
        full_index = pd.MultiIndex.from_product(
            [df["REGION"].unique(), df["BRAND"].unique(), all_dates],
            names=["REGION", "BRAND", "DATE"]
        )
        df = df.set_index(["REGION", "BRAND", "DATE"]).reindex(full_index, fill_value=0).reset_index()
    
        df["MONTH_NUM"] = df["DATE"].dt.month
        df["YEAR_NUM"] = df["DATE"].dt.year
        df["LAG_1"] = df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].shift(1)
        df["LAG_2"] = df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].shift(2)
        df["LAG_3"] = df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].shift(3)
        df["ROLLING_MEAN_3"] = df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df["ROLLING_MEAN_6"] = df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].transform(lambda x: x.rolling(6, min_periods=1).mean())
        df["MONTH_SIN"] = np.sin(2 * np.pi * df["MONTH_NUM"] / 12)
        df["MONTH_COS"] = np.cos(2 * np.pi * df["MONTH_NUM"] / 12)
    
        Q1 = df["NET_SALE_AMOUNT"].quantile(0.01)
        Q3 = df["NET_SALE_AMOUNT"].quantile(0.99)
        IQR = Q3 - Q1
        df = df[(df["NET_SALE_AMOUNT"] >= Q1 - 1.5 * IQR) & (df["NET_SALE_AMOUNT"] <= Q3 + 1.5 * IQR)]
        df = df.dropna(subset=["LAG_1", "LAG_2", "LAG_3"])
        df["LOG_SALES"] = np.log1p(df["NET_SALE_AMOUNT"])
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
    
        brand_encoder = LabelEncoder()
        df["BRAND_ENC"] = brand_encoder.fit_transform(df["BRAND"])
        return df, brand_encoder
    
    def train_xgboost_model(X_train, y_train):
        param_grid = {
            'n_estimators': [100],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1],
            'colsample_bytree': [0.8, 1]
        }
        model = XGBRegressor(random_state=42, eval_metric='mae')
        grid = GridSearchCV(model, param_grid, cv=TimeSeriesSplit(n_splits=3), scoring='neg_mean_absolute_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        return grid.best_estimator_, grid.best_params_
    
    df, brand_encoder = prepare_features(raw_df)
    features = ["MONTH_NUM", "LAG_1", "LAG_2", "LAG_3", "ROLLING_MEAN_3", "ROLLING_MEAN_6", "BRAND_ENC", "MONTH_SIN", "MONTH_COS"]
    
    # Overall aggregation for combined graphs and test comparison
    overall_df = df.groupby(["BRAND", "DATE"]).agg({
        "NET_SALE_AMOUNT": "sum",
        "MONTH_NUM": "first",
        "YEAR_NUM": "first",
        "LAG_1": "sum",
        "LAG_2": "sum",
        "LAG_3": "sum",
        "ROLLING_MEAN_3": "mean",
        "ROLLING_MEAN_6": "mean",
        "BRAND_ENC": "first",
        "MONTH_SIN": "first",
        "MONTH_COS": "first"
    }).reset_index()
    overall_df["REGION"] = "Overall"
    overall_df["LOG_SALES"] = np.log1p(overall_df["NET_SALE_AMOUNT"])
    
    # Train model
    train_overall = overall_df[overall_df["YEAR_NUM"] < 2025]
    test_overall = overall_df[(overall_df["YEAR_NUM"] == 2025) & (overall_df["DATE"].dt.month <= 4)].copy()
    X_train_overall = train_overall[features]
    y_train_overall = train_overall["LOG_SALES"]
    best_model_overall, _ = train_xgboost_model(X_train_overall, y_train_overall)
    
    # Predict test period
    test_overall["PREDICTED_SALES"] = np.expm1(best_model_overall.predict(test_overall[features])).round(0).astype(int)
    test_overall["TYPE"] = "Test Prediction"
    
    # Actual test sales for the same period
    actual_test_sales = overall_df[(overall_df["YEAR_NUM"] == 2025) & (overall_df["DATE"].dt.month <= 4)][["DATE", "REGION", "BRAND", "NET_SALE_AMOUNT"]].copy()
    actual_test_sales.rename(columns={"NET_SALE_AMOUNT": "ACTUAL_SALES"}, inplace=True)
    
    # Prepare test comparison DataFrame
    test_comparison_df = pd.merge(
        actual_test_sales,
        test_overall[["DATE", "REGION", "BRAND", "PREDICTED_SALES"]],
        on=["DATE", "REGION", "BRAND"],
        how="inner"
    )
    
    # Forecast Apr-Dec 2025 overall (for graphs)
    forecast_results_overall = []
    overall_forecast_df = overall_df.copy()
    for forecast_date in pd.date_range("2025-04-01", "2025-12-01", freq="MS"):
        temp_df = overall_forecast_df[overall_forecast_df["DATE"] == forecast_date].copy()
        if temp_df.empty:
            continue
        predicted_sales = np.expm1(best_model_overall.predict(temp_df[features])).round(0).astype(int)
        temp_df["SALES"] = predicted_sales
        temp_df["TYPE"] = "Forecast Apr-Dec 2025"
        forecast_results_overall.append(temp_df[["DATE", "REGION", "BRAND", "SALES", "TYPE"]])
    
        overall_forecast_df.loc[overall_forecast_df["DATE"] == forecast_date, "NET_SALE_AMOUNT"] = predicted_sales
        overall_forecast_df["LAG_1"] = overall_forecast_df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].shift(1)
        overall_forecast_df["LAG_2"] = overall_forecast_df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].shift(2)
        overall_forecast_df["LAG_3"] = overall_forecast_df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].shift(3)
        overall_forecast_df["ROLLING_MEAN_3"] = overall_forecast_df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        overall_forecast_df["ROLLING_MEAN_6"] = overall_forecast_df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].transform(lambda x: x.rolling(6, min_periods=1).mean())
    
    forecast_df_overall = pd.concat(forecast_results_overall)
    combined_overall = pd.concat([
        overall_df[overall_df["YEAR_NUM"] < 2025][["DATE", "REGION", "BRAND", "NET_SALE_AMOUNT"]].rename(columns={"NET_SALE_AMOUNT": "SALES"}).assign(TYPE="Historical"),
        test_overall[["DATE", "REGION", "BRAND", "PREDICTED_SALES"]].rename(columns={"PREDICTED_SALES": "SALES"}).assign(TYPE="Test Prediction"),
        forecast_df_overall
    ]).reset_index(drop=True)
    
    # === Per-region forecasts ===
    region_dfs = {}
    for region in df["REGION"].unique():
        region_df = df[df["REGION"] == region]
        train = region_df[region_df["YEAR_NUM"] < 2025]
        test = region_df[(region_df["YEAR_NUM"] == 2025) & (region_df["DATE"].dt.month <= 4)].copy()
    
        X_train = train[features]
        y_train = train["LOG_SALES"]
        best_model, _ = train_xgboost_model(X_train, y_train)
    
        # Actual test sales for region
        actual_test_sales_region = region_df[(region_df["YEAR_NUM"] == 2025) & (region_df["DATE"].dt.month <= 4)][["DATE", "REGION", "BRAND", "NET_SALE_AMOUNT"]].copy()
        actual_test_sales_region.rename(columns={"NET_SALE_AMOUNT": "ACTUAL_SALES"}, inplace=True)
    
        test = test.copy()
        test["PREDICTED_SALES"] = np.expm1(best_model.predict(test[features])).round(0).astype(int)
    
        # Merge actual and predicted for test comparison table
        test_comparison_region = pd.merge(
            actual_test_sales_region,
            test[["DATE", "REGION", "BRAND", "PREDICTED_SALES"]],
            on=["DATE", "REGION", "BRAND"],
            how="inner"
        )
    
        # Forecast Apr-Dec 2025 for region (for graphs)
        forecast_results = []
        region_forecast_df = region_df.copy()
        for forecast_date in pd.date_range("2025-04-01", "2025-12-01", freq="MS"):
            temp_df = region_forecast_df[region_forecast_df["DATE"] == forecast_date].copy()
            if temp_df.empty:
                continue
            predicted_sales = np.expm1(best_model.predict(temp_df[features])).round(0).astype(int)
            temp_df["SALES"] = predicted_sales
            temp_df["TYPE"] = "Forecast Apr-Dec 2025"
            forecast_results.append(temp_df[["DATE", "REGION", "BRAND", "SALES", "TYPE"]])
    
            region_forecast_df.loc[region_forecast_df["DATE"] == forecast_date, "NET_SALE_AMOUNT"] = predicted_sales
            region_forecast_df["LAG_1"] = region_forecast_df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].shift(1)
            region_forecast_df["LAG_2"] = region_forecast_df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].shift(2)
            region_forecast_df["LAG_3"] = region_forecast_df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].shift(3)
            region_forecast_df["ROLLING_MEAN_3"] = region_forecast_df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].transform(lambda x: x.rolling(3, min_periods=1).mean())
            region_forecast_df["ROLLING_MEAN_6"] = region_forecast_df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].transform(lambda x: x.rolling(6, min_periods=1).mean())
    
        forecast_df = pd.concat(forecast_results)
        combined_region = pd.concat([
            region_df[region_df["YEAR_NUM"] < 2025][["DATE", "REGION", "BRAND", "NET_SALE_AMOUNT"]].rename(columns={"NET_SALE_AMOUNT": "SALES"}).assign(TYPE="Historical"),
            test[["DATE", "REGION", "BRAND", "PREDICTED_SALES"]].rename(columns={"PREDICTED_SALES": "SALES"}).assign(TYPE="Test Prediction"),
            forecast_df
        ]).reset_index(drop=True)
    
        # Store both dataframes for this region
        region_dfs[region] = {
            "forecast_df": combined_region,
            "test_comparison_df": test_comparison_region
        }
    
    # Flatten the return dictionary so each key maps directly to a DataFrame
    flattened_result = {
        "overall_forecast_df": combined_overall,
        "overall_test_comparison_df": test_comparison_df,
    }
    for region, dfs in region_dfs.items():
        flattened_result[f"{region}_forecast_df"] = dfs["forecast_df"]
        flattened_result[f"{region}_test_comparison_df"] = dfs["test_comparison_df"]
    
    return flattened_result
