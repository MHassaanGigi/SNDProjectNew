# app/routes.py
from flask import Blueprint, render_template, jsonify, request
import numpy as np
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def home():
    return render_template('home.html')  # This will extend base.html

@main_bp.route('/about')
def about():
    return render_template('about.html')  

@main_bp.route('/main')
def main():
    return render_template('main.html')

@main_bp.route('/SkuWise')
def SkuWise():
    return render_template('SkuWise.html')

@main_bp.route('/RegionWise')
def RegionWise():
    return render_template('RegionWise.html')

@main_bp.route('/TopBrands')
def TopBrands():
    return render_template('TopBrands.html')

from SWM import run_full_forecast  # updated import for new merged function
import pandas as pd
import traceback
import logging

@main_bp.route('/run-forecasting', methods=['GET'])
def run_forecasting_route():
    try:
        logging.error("DEBUG: Entered /run-forecasting endpoint.")

        # Unpack both DataFrames
        comparison_df, monthly_df,sku_df,town_df = run_full_forecast(debug=False)

        logging.error("DEBUG: run_forecasting() returned comparison_df columns: %s", comparison_df.columns.tolist())
        logging.error("DEBUG: run_forecasting() returned monthly_df columns: %s", monthly_df.columns.tolist())

        # Function to round numeric columns and convert to dict
        def df_to_records(df):
            for col in df.select_dtypes(include=[float, int]):
                df[col] = df[col].round().astype(int)
            return df.to_dict(orient='records')

        # Convert both
        comparison_records = df_to_records(comparison_df)
        monthly_records = df_to_records(monthly_df)
        sku_records = df_to_records(sku_df)
        town_records = df_to_records(town_df)

        # Return both as separate keys
        return jsonify({
            "comparison": comparison_records,
            "monthly": monthly_records,
            "SkuRecords": sku_records,
            "TownRecords": town_records
        })

    except Exception as e:
        logging.error("❌ Exception in /run-forecasting: %s", str(e))
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

from Region import run_forecast

@main_bp.route('/region-forecast',methods=['GET'])
def forecast_route():
    division = request.args.get('division', None)
    combined_df, test_r2 = run_forecast(division=division)
    # Convert DataFrame to records for JSON response
    data_records = combined_df[["REGION", "DATE","DATE_LABEL","MONTHLY_SALES", "PREDICTED_SALES", "TYPE"]].copy().replace([np.nan, np.inf, -np.inf], None)
    # Convert datetime to string for JSON serialization
    data_records["DATE"] = data_records["DATE"].dt.strftime("%Y-%m-%d")
    result_json = data_records.to_dict(orient='records')

    return jsonify({
        "data": result_json,
        "test_r2": test_r2
    })

from TopBrands import get_forecast_and_test_data
@main_bp.route('/TopBrands-Forecast', methods=['GET'])
def TopBrands_forecast():
    division = request.args.get("division", None)

    def process_forecast(region):
        # Get forecast/test data
        table_df, graph_df = get_forecast_and_test_data(region, division)

        # --- Fix actual sales table ---
        if "ACTUAL_SALES" in table_df.columns:
            table_df["TYPE"] = "Actual_vs_Predicted"
            table_df.rename(columns={"ACTUAL_SALES": "MONTHLY_SALES"}, inplace=True)

        # --- Fix forecast dataframe ---
        if "SALES" in graph_df.columns:
            graph_df.rename(columns={"SALES": "PREDICTED_SALES"}, inplace=True)

        graph_df = graph_df.sort_values(["BRAND", "DATE"])  # optional: ensure sorted
        #graph_df = graph_df.drop_duplicates(subset="DATE", keep="first")
        
        return table_df, graph_df

    # Process both regions
    north_table, north_graph = process_forecast("NORTH")
    south_table, south_graph = process_forecast("SOUTH")

    # Return combined response
    return jsonify({
        "NORTH": {
            "table": north_table.to_dict(orient="records"),
            "graph": north_graph.to_dict(orient="records")
        },
        "SOUTH": {
            "table": south_table.to_dict(orient="records"),
            "graph": south_graph.to_dict(orient="records")
        }
    })
