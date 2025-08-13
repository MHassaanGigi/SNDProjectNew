import pandas as pd
import numpy as np
from db_connect import run_query
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
def get_forecast_and_test_data(var_region=None, var_division=None):
    
    if var_division is None:
        var_division = 'CANDYLAND'
    
    base_query = """
    SELECT REGION, DIVISION, BRAND, YEAR, MONTH, SUM(NET_SALE_AMOUNT) AS NET_SALE_AMOUNT
    FROM FINAL_QUERY
    """
    conditions = []
    if var_region:
        conditions.append(f"REGION = '{var_region}'")
    if var_division:
        conditions.append(f"DIVISION = '{var_division}'")
    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)
    base_query += """
    GROUP BY REGION, DIVISION, BRAND, YEAR, MONTH
    ORDER BY REGION, DIVISION, BRAND, YEAR, MONTH
    """
    raw_df = pd.DataFrame(run_query(base_query))

    # === Top 3 Brands Filter ===
    if var_region or var_division:
        # Filter top 3 brands globally within the filtered data
        top_brands = raw_df.groupby("BRAND")['NET_SALE_AMOUNT'].sum().reset_index()
        top_3_brands = top_brands.sort_values("NET_SALE_AMOUNT", ascending=False).head(3)["BRAND"].tolist()
        raw_df = raw_df[raw_df["BRAND"].isin(top_3_brands)]
    else:
        # No filters, top 3 brands per (REGION, DIVISION) group
        top_brands = raw_df.groupby(["REGION", "DIVISION", "BRAND"])['NET_SALE_AMOUNT'].sum().reset_index()
        top_3_brands_per_group = top_brands.sort_values(
            ["REGION", "DIVISION", "NET_SALE_AMOUNT"],
            ascending=[True, True, False]
        ).groupby(["REGION", "DIVISION"]).head(3)
        raw_df = raw_df.merge(top_3_brands_per_group[["REGION", "DIVISION", "BRAND"]],
                            on=["REGION", "DIVISION", "BRAND"])
        
        print(f"Top 3 brands filtered (region={var_region}, division={var_division}):", 
         top_3_brands if (var_region or var_division) else top_3_brands_per_group)
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

        #Q1 = df["NET_SALE_AMOUNT"].quantile(0.01)
        #Q3 = df["NET_SALE_AMOUNT"].quantile(0.99)
        #IQR = Q3 - Q1
        #df = df[(df["NET_SALE_AMOUNT"] >= Q1 - 1.5 * IQR) & (df["NET_SALE_AMOUNT"] <= Q3 + 1.5 * IQR)]
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
        return grid.best_estimator_

    df, brand_encoder = prepare_features(raw_df)
    features = ["MONTH_NUM", "LAG_1", "LAG_2", "LAG_3", "ROLLING_MEAN_3", "ROLLING_MEAN_6", "BRAND_ENC", "MONTH_SIN", "MONTH_COS"]
    
    regions = [var_region] if var_region else list(pd.unique(df["REGION"]))

    combined_forecast_results = []
    combined_test_results = []

    for region in regions:
        region_df = df[df["REGION"] == region]
        train = region_df[region_df["YEAR_NUM"] < 2025]
        test = region_df[(region_df["YEAR_NUM"] == 2025) & (region_df["DATE"].dt.month <= 4)].copy()

        X_train = train[features]
        y_train = train["LOG_SALES"]
        best_model = train_xgboost_model(X_train, y_train)

        # Predict test period (Jan-Apr 2025)
        test["PREDICTED_SALES"] = np.expm1(best_model.predict(test[features])).round(0).astype(int)
        test_vs_pred = test[["DATE", "REGION", "BRAND", "NET_SALE_AMOUNT", "PREDICTED_SALES"]].copy()
        test_vs_pred.rename(columns={"NET_SALE_AMOUNT": "ACTUAL_SALES"}, inplace=True)
        combined_test_results.append(test_vs_pred)

        # Forecast Apr-Dec 2025
        region_forecast_df = region_df.copy()
        forecast_results = []
        for forecast_date in pd.date_range("2025-04-01", "2025-12-01", freq="MS"):
            temp_df = region_forecast_df[region_forecast_df["DATE"] == forecast_date].copy()
            if temp_df.empty:
                continue
            predicted_sales = np.expm1(best_model.predict(temp_df[features])).round(0).astype(int)
            temp_df["SALES"] = predicted_sales
            temp_df["TYPE"] = "Forecast Apr-Dec 2025"
            forecast_results.append(temp_df[["DATE", "REGION", "BRAND", "SALES", "TYPE"]])

            region_forecast_df.loc[region_forecast_df["DATE"] == forecast_date, "NET_SALE_AMOUNT"] = predicted_sales
            # update lag and rolling features for next forecast step
            region_forecast_df["LAG_1"] = region_forecast_df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].shift(1)
            region_forecast_df["LAG_2"] = region_forecast_df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].shift(2)
            region_forecast_df["LAG_3"] = region_forecast_df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].shift(3)
            region_forecast_df["ROLLING_MEAN_3"] = region_forecast_df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].transform(lambda x: x.rolling(3, min_periods=1).mean())
            region_forecast_df["ROLLING_MEAN_6"] = region_forecast_df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].transform(lambda x: x.rolling(6, min_periods=1).mean())

        if forecast_results:
            combined_forecast_results.append(pd.concat(forecast_results))

    # Combine results across regions (if multiple regions passed)
    test_vs_pred_df = pd.concat(combined_test_results).reset_index(drop=True)
    forecast_df = pd.concat(combined_forecast_results).reset_index(drop=True) if combined_forecast_results else pd.DataFrame()

    # Prepare forecast_df to match test_vs_pred_df columns
    forecast_df_renamed = forecast_df.rename(columns={"SALES": "PREDICTED_SALES"})
    forecast_df_renamed["ACTUAL_SALES"] = np.nan  # No actual sales for future months

    # Concatenate Jan–Apr (test) with Apr–Dec (forecast) to get full 2025
    full_2025_df = pd.concat([test_vs_pred_df, forecast_df_renamed], ignore_index=True)

    # Optional: sort by date and brand
    full_2025_df = full_2025_df.sort_values(["DATE", "REGION", "BRAND"]).reset_index(drop=True)

    # Fill missing TYPE values with "Forecast Apr-Dec 2025"
    full_2025_df["TYPE"] = full_2025_df["TYPE"].fillna("Forecast Apr-Dec 2025")

    # Drop ACTUAL_SALES column if not needed
    if "ACTUAL_SALES" in full_2025_df.columns:
        full_2025_df = full_2025_df.drop(columns=["ACTUAL_SALES"])

    full_2025_df = full_2025_df.sort_values(["BRAND", "DATE"]).reset_index(drop=True)
    return test_vs_pred_df, full_2025_df
