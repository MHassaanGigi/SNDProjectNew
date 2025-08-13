import pandas as pd
import numpy as np
from db_connect import run_query
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def generate_sales_forecast(selected_division, selected_area):
    # === SQL Query ===
    query = f"""
    SELECT
        DIVISION,
        AREA,
        FLAVOUR,
        YEAR,
        MONTH,
        SUM(NET_SALE_AMOUNT) AS NET_SALE_AMOUNT
    FROM FINAL_QUERY
    WHERE
        YEAR BETWEEN 2023 AND 2025
        AND FLAVOUR IS NOT NULL
        AND DIVISION = '{selected_division}'
        AND AREA = '{selected_area}'
    GROUP BY
        DIVISION, AREA, FLAVOUR, YEAR, MONTH
    ORDER BY
        DIVISION, AREA, FLAVOUR, YEAR, MONTH;
    """

    df = pd.DataFrame(run_query(query))
    df["FLAVOUR"] = df["FLAVOUR"].str.upper().str.strip()

    # Top 3 Flavours filter
    top_flavours = (
        df.groupby("FLAVOUR")["NET_SALE_AMOUNT"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )
    df = df[df["FLAVOUR"].isin(top_flavours)]

    # Prepare features function inside
    def prepare_flavour_features(df):
        df["DATE"] = pd.to_datetime(df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str).str.zfill(2) + "-01")
        df = df.sort_values(["AREA", "FLAVOUR", "DATE"])
        df["NET_SALE_AMOUNT"] = pd.to_numeric(df["NET_SALE_AMOUNT"], errors="coerce").fillna(0)
        df.loc[df["NET_SALE_AMOUNT"] < 0, "NET_SALE_AMOUNT"] = 0

        all_dates = pd.date_range("2023-01-01", "2025-12-01", freq="MS")
        full_index = pd.MultiIndex.from_product(
            [df["AREA"].unique(), df["FLAVOUR"].unique(), all_dates],
            names=["AREA", "FLAVOUR", "DATE"]
        )
        df = df.set_index(["AREA", "FLAVOUR", "DATE"]).reindex(full_index, fill_value=0).reset_index()

        df["MONTH_NUM"] = df["DATE"].dt.month
        df["YEAR_NUM"] = df["DATE"].dt.year

        df["LAG_1"] = df.groupby(["AREA", "FLAVOUR"])["NET_SALE_AMOUNT"].shift(1)
        df["LAG_2"] = df.groupby(["AREA", "FLAVOUR"])["NET_SALE_AMOUNT"].shift(2)
        df["LAG_3"] = df.groupby(["AREA", "FLAVOUR"])["NET_SALE_AMOUNT"].shift(3)
        df["ROLLING_MEAN_3"] = df.groupby(["AREA", "FLAVOUR"])["NET_SALE_AMOUNT"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df["ROLLING_MEAN_6"] = df.groupby(["AREA", "FLAVOUR"])["NET_SALE_AMOUNT"].transform(lambda x: x.rolling(6, min_periods=1).mean())

        df["MONTH_SIN"] = np.sin(2 * np.pi * df["MONTH_NUM"] / 12)
        df["MONTH_COS"] = np.cos(2 * np.pi * df["MONTH_NUM"] / 12)

        Q1 = df["NET_SALE_AMOUNT"].quantile(0.01)
        Q3 = df["NET_SALE_AMOUNT"].quantile(0.99)
        df["NET_SALE_AMOUNT"] = np.clip(df["NET_SALE_AMOUNT"], Q1, Q3)

        df["LOG_SALES"] = np.log1p(df["NET_SALE_AMOUNT"])
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        flavour_encoder = LabelEncoder()
        df["FLAVOUR_ENC"] = flavour_encoder.fit_transform(df["FLAVOUR"])
        return df, flavour_encoder

    df, flavour_encoder = prepare_flavour_features(df)

    features = ["MONTH_NUM", "LAG_1", "LAG_2", "LAG_3", "ROLLING_MEAN_3",
                "ROLLING_MEAN_6", "FLAVOUR_ENC", "MONTH_SIN", "MONTH_COS"]

    forecast_rows = []
    metrics_summary = []

    for (AREA, flavour), group in df.groupby(["AREA", "FLAVOUR"]):
        group = group.sort_values("DATE")
        X = group[features]
        y = group["LOG_SALES"]

        X_train = X[group["YEAR_NUM"] < 2025]
        y_train = y[group["YEAR_NUM"] < 2025]

        X_pred_early = X[(group["YEAR_NUM"] == 2025) & (group["DATE"].dt.month <= 4)]
        dates_pred_early = group.loc[X_pred_early.index, "DATE"]

        model = XGBRegressor(objective='reg:squarederror', n_estimators=300, learning_rate=0.05, max_depth=5)
        model.fit(X_train, y_train)

        y_pred_early_log = model.predict(X_pred_early)
        y_pred_early = np.expm1(y_pred_early_log)
        y_actual_early = group.loc[X_pred_early.index, "NET_SALE_AMOUNT"]

        mae = mean_absolute_error(y_actual_early, y_pred_early)
        r2 = r2_score(y_actual_early, y_pred_early)
        metrics_summary.append({"AREA": AREA, "FLAVOUR": flavour, "MAE": mae, "R2": r2})

        for date, sales in zip(dates_pred_early, y_pred_early):
            forecast_rows.append({"AREA": AREA, "FLAVOUR": flavour, "DATE": date, "SALES": sales, "TYPE": "Predicted 2025"})

        last_known = group.copy()
        for month in range(5, 13):
            forecast_date = pd.to_datetime(f"2025-{month:02d}-01")
            past_data = last_known[last_known["DATE"] < forecast_date].sort_values("DATE")

            lag_1 = past_data.iloc[-1]["NET_SALE_AMOUNT"]
            lag_2 = past_data.iloc[-2]["NET_SALE_AMOUNT"]
            lag_3 = past_data.iloc[-3]["NET_SALE_AMOUNT"]
            rolling_3 = past_data["NET_SALE_AMOUNT"].tail(3).mean()
            rolling_6 = past_data["NET_SALE_AMOUNT"].tail(6).mean()
            flavour_enc = flavour_encoder.transform([flavour])[0]
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            feature_vector = pd.DataFrame([{
                "MONTH_NUM": month, "LAG_1": lag_1, "LAG_2": lag_2, "LAG_3": lag_3,
                "ROLLING_MEAN_3": rolling_3, "ROLLING_MEAN_6": rolling_6,
                "FLAVOUR_ENC": flavour_enc, "MONTH_SIN": month_sin, "MONTH_COS": month_cos
            }])
            pred_sales = np.expm1(model.predict(feature_vector)[0])
            forecast_rows.append({"AREA": AREA, "FLAVOUR": flavour, "DATE": forecast_date, "SALES": pred_sales, "TYPE": "Forecast 2025"})

            last_known = pd.concat([last_known, pd.DataFrame([{
                "AREA": AREA, "FLAVOUR": flavour, "DATE": forecast_date, "NET_SALE_AMOUNT": pred_sales
            }])], ignore_index=True)

        # Add actuals for 2024
        actual_2024 = group[(group["YEAR_NUM"] == 2024)][["DATE", "NET_SALE_AMOUNT"]]
        for _, row in actual_2024.iterrows():
            forecast_rows.append({"AREA": AREA, "FLAVOUR": flavour, "DATE": row["DATE"], "SALES": row["NET_SALE_AMOUNT"], "TYPE": "Actual 2024"})

    forecast_df = pd.DataFrame(forecast_rows)

    # Optionally return metrics summary as well
    metrics_df = pd.DataFrame(metrics_summary)

    return forecast_df, metrics_df
