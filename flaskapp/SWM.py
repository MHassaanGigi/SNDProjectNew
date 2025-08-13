import pandas as pd
from db_connect import run_query

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
    AND MONTH <= 4
    AND TOWN = 'KARACHI'
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

# Clean column names
df.columns = df.columns.str.strip().str.upper()

# Create DATE column (monthly granularity)
df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str))

# Sort for lag calculation
df = df.sort_values(by=['SKU_CODE', 'TOWN', 'DATE'])

import numpy as np
#Capping Outliers for better generalization 
lower = df['TARGET'].quantile(0.01)
upper = df['TARGET'].quantile(0.99)
df['TARGET_W'] = df['TARGET'].clip(lower, upper)

# Add a small constant to avoid log(0)
df['TARGET_LOG'] = np.log1p(df['TARGET_W'])

# Lag features (monthly)
df['LAG_1'] = df.groupby(['SKU_CODE', 'TOWN'])['TARGET_LOG'].shift(1)
df['LAG_2'] = df.groupby(['SKU_CODE', 'TOWN'])['TARGET_LOG'].shift(2)
df['LAG_3'] = df.groupby(['SKU_CODE', 'TOWN'])['TARGET_LOG'].shift(3)

# Define rolling features
df['ROLL_MEAN_3'] = (df.groupby(['SKU_CODE', 'TOWN'])['TARGET_LOG'].transform(lambda x: x.shift(1).rolling(window=3).mean()))

df['ROLL_MEAN_6'] = (df.groupby(['SKU_CODE', 'TOWN'])['TARGET_LOG'].transform(lambda x: x.shift(1).rolling(window=6).mean()))

df['ROLL_STD_3'] = (df.groupby(['SKU_CODE', 'TOWN'])['TARGET_LOG'].transform(lambda x: x.shift(1).rolling(window=3).std()))

# Define growth metrics
df['YOY_GROWTH'] = (df.groupby(['SKU_CODE', 'TOWN'])['TARGET_LOG'].transform(lambda x: x.pct_change(periods=12)))

df['MOM_GROWTH'] = (df.groupby(['SKU_CODE', 'TOWN'])['TARGET_LOG'].transform(lambda x: x.pct_change(periods=1)))

# Drop rows with missing values after feature creation
df = df.dropna().reset_index(drop=True)

import category_encoders as ce

# Target encode SKU_CODE
encoder = ce.TargetEncoder(cols=['SKU_CODE'])
df['SKU_CODE_ENC'] = encoder.fit_transform(df['SKU_CODE'], df['TARGET_LOG'])

# Count encode TOWN
df['TOWN_ENC'] = df['TOWN'].map(df['TOWN'].value_counts())

from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

def train_xgboost(X_train, y_train, X_test, y_test):
    X_train = pd.DataFrame(X_train).replace([np.inf, -np.inf], np.nan).dropna()
    X_test = pd.DataFrame(X_test).replace([np.inf, -np.inf], np.nan).dropna()

    # Align y with X after dropping NaNs
    y_train = pd.Series(y_train).loc[X_train.index]
    y_test = pd.Series(y_test).loc[X_test.index]

    # Define XGBoost model
    model = XGBRegressor(
        learning_rate=0.1,
        max_depth=3,
        n_estimators=300,
        subsample=1.0,
        random_state=42,
        objective='reg:squarederror'
    )

    # Fit model with early stopping
    model.fit(X_train, y_train,eval_set=[(X_test, y_test)],early_stopping_rounds=20,verbose=False)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    rmse = root_mean_squared_error(y_test, y_pred, squared=False)  # Proper RMSE
    r2 = r2_score(y_test, y_pred)

    return {"model": model,"rmse": rmse,"r2": r2,"y_pred": y_pred}

def generate_rolling_features(df):
    """
    Splits the dataset into fixed-date training and testing windows.
    Train: Jan 2024 – Feb 2025
    Test: Mar 2025 – Apr 2025
    """

    # Fixed date boundaries
    train_start_year, train_start_month = 2024, 1
    train_end_year, train_end_month = 2025, 2
    test_start_year, test_start_month = 2025, 3
    test_end_year, test_end_month = 2025, 4

    # Copy to avoid mutating original DataFrame
    df = df.copy()

    # Sort to ensure proper rolling behavior
    df = df.sort_values(by=['SKU_CODE', 'TOWN', 'YEAR', 'MONTH'])

    # Training set mask
    train_mask = (
        ((df['YEAR'] > train_start_year) | ((df['YEAR'] == train_start_year) & (df['MONTH'] >= train_start_month))) &
        ((df['YEAR'] < train_end_year) | ((df['YEAR'] == train_end_year) & (df['MONTH'] <= train_end_month)))
    )
    train_rolling = df[train_mask]

    # Test set mask
    test_mask = (
        ((df['YEAR'] > test_start_year) | ((df['YEAR'] == test_start_year) & (df['MONTH'] >= test_start_month))) &
        ((df['YEAR'] < test_end_year) | ((df['YEAR'] == test_end_year) & (df['MONTH'] <= test_end_month)))
    )
    test_rolling = df[test_mask]

    return train_rolling, test_rolling
