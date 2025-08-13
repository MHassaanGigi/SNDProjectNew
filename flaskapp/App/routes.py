# app/routes.py
from flask import Blueprint, render_template

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

from flask import jsonify
from SkuWuseModel import run_forecasting,generate_forward_forecast
import numpy as np
trained_model_cache = None

@main_bp.route('/run-forecasting', methods=['GET'])
def run_forecasting_route():
    global trained_model_cache
    try:
        monthly_df,trained_model_cache = run_forecasting() 
        for col in monthly_df.select_dtypes(include=[np.number]).columns:
            monthly_df[col] = monthly_df[col].round().astype(int)
        results_list = monthly_df.to_dict(orient="records")  # convert DF to list of dicts
        return jsonify(results_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@main_bp.route('/generate-forward-forecast', methods=['GET'])
def forward_forecast_route():
    global trained_model_cache
    try:
        if trained_model_cache is None:
            print("No cached models, running training now...")
        _, trained_model_cache = run_forecasting()
        print("Training completed:", trained_model_cache)
        if trained_model_cache is None:
            return jsonify({"error": "Failed to train models"}), 500
            
        forecast_df = generate_forward_forecast(trained_model_cache)  # uses hardcoded territory for now
        for col in forecast_df.select_dtypes(include=[np.number]).columns:
            forecast_df[col] = forecast_df[col].round().astype(int)
        results_list = forecast_df.to_dict(orient="records")
        return jsonify(results_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

from Region import run_forecast
from flask import request
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
