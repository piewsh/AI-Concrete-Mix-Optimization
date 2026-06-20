import os
import base64
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from flask import Flask, render_template, request, jsonify, send_file, session
from io import BytesIO
from datetime import datetime
import pytz
from tempfile import NamedTemporaryFile
from sklearn.preprocessing import MinMaxScaler
from platypus import NSGAII, Problem, Real
from fpdf import FPDF
import torch
from predict import *
from torch_geometric.data import Data

# Custom module imports
from concMixIS import (
    target_strength_calculation,
    water_cement_ratio_calculation,
    water_content_calculation,
    cement_content_calculation,
    vol_of_CAnFA_calculation,
    mix_calculation,
    fly_cement_content_calculation,
    fly_mix_calculation
)
from optimize import optimize_mix_logic
from pdfMaker import create_mix_pdf, create_optimization_pdf

# Model imports (ensure these exist)
from model import ConcreteGNN
from trainann import ConcreteANN

app = Flask(__name__)
app.secret_key = "YOUR_SECRET_KEY"  

#############################################
# Helper: Load CSV Data
#############################################
def load_mix_data():
    data_path = './OptData.csv'
    df = pd.read_csv(data_path)
    df.columns = [col.strip() for col in df.columns]
    return df

#############################################
# Concrete Optimizer Class
#############################################
class ConcreteOptimizer:
    def __init__(self, models_path, min_max_values, feature_columns, desired_strength, expected_date):
        self.models = self.load_models(models_path)
        self.min_max_values = min_max_values
        self.feature_columns = feature_columns
        self.desired_strength = desired_strength
        self.expected_date = expected_date

    @staticmethod
    def load_models(path):
        model_names = ["Strength", "CO2", "Cost"]
        models = {}
        for name in model_names:
            filepath = os.path.join(path, f"XGB_{name}_model.pkl")
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")
            models[name] = joblib.load(filepath)
        return models

    def optimize_concrete_mix(self, x):
        input_features = list(x) + [self.expected_date]
        strength_pred = self.models["Strength"].predict([input_features])[0]
        co2_pred = self.models["CO2"].predict([input_features])[0]
        cost_pred = self.models["Cost"].predict([input_features])[0]
        obj_strength_dev = abs(strength_pred - self.desired_strength)
        return [obj_strength_dev, co2_pred, cost_pred], [self.desired_strength - strength_pred]

    def run_optimization(self, iterations):
        problem = Problem(10, 3, 1)
        problem.types[:] = [
            Real(self.min_max_values.loc[feat, "min"],
            self.min_max_values.loc[feat, "max"]) for feat in self.feature_columns
        ]
        problem.function = self.optimize_concrete_mix
        problem.constraints[:] = "<=0"

        algo = NSGAII(problem)
        algo.run(iterations)
        feasible = [s for s in algo.result if s.feasible]
        if not feasible:
            return np.array([]), np.array([]), np.array([])
        optimal_solutions = np.array([[s.objectives[0], s.objectives[1], s.objectives[2]] for s in feasible])
        optimal_features = np.array([s.variables for s in feasible])
        scaler = MinMaxScaler()
        scaled_optimal_solutions = scaler.fit_transform(optimal_solutions)
        return optimal_solutions, scaled_optimal_solutions, optimal_features

#############################################
# Flask Routes
#############################################
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/mix_optimization", methods=["GET", "POST"])
def mix_optimization_route():
    df = load_mix_data()
    required_cols = ["Cement", "Clinker", "Slag", "FlyAsh", "Limestone", "Gypsum",
                     "Water", "SuperPlasticizer", "CoarseAggregate", "FineAggregate"]
    # Validate CSV columns
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in CSV.")
    data = df[required_cols].copy()
    min_max_values = data.describe().loc[["min", "max"]].transpose()
    # Default slider values = min from CSV
    default_values = {col: float(min_max_values.loc[col, "min"]) for col in required_cols}
    default_values["desired_strength"] = 30
    default_values["expected_date"] = 28
    default_values["iterations"] = 1000

    if request.method == "POST":
        try:
            # Retrieve form inputs
            cement = float(request.form["Cement"])
            clinker = float(request.form["Clinker"])
            slag = float(request.form["Slag"])
            flyash = float(request.form["FlyAsh"])
            limestone = float(request.form["Limestone"])
            gypsum = float(request.form["Gypsum"])
            water = float(request.form["Water"])
            super_plasticizer = float(request.form["SuperPlasticizer"])
            coarse_agg = float(request.form["CoarseAggregate"])
            fine_agg = float(request.form["FineAggregate"])
            desired_strength = float(request.form["desired_strength"])
            expected_date = float(request.form["expected_date"])
            iterations = int(request.form["iterations"])
        except Exception as e:
            return f"Error in input parameters: {e}", 400

        optimizer = ConcreteOptimizer(
            models_path="./Mod",
            min_max_values=min_max_values,
            feature_columns=required_cols,
            desired_strength=desired_strength,
            expected_date=expected_date
        )
        optimal_solutions, scaled_optimal_solutions, optimal_features = optimizer.run_optimization(iterations)

        if optimal_solutions.size == 0:
            # No feasible solution
            return render_template("mix_optimization.html",
                                   form_data=request.form,
                                   results=None,
                                   plot_url=None,
                                   plotly_html=None,
                                   pdf_base64=None,
                                   obj_df="",
                                   mix_df="")

        # Build results dict
        results = {
            "best_strength_dev": float(optimal_solutions[np.argmin(optimal_solutions[:, 0]), 0]),
            "strength_solution": optimal_solutions[np.argmin(optimal_solutions[:, 0])].tolist(),
            "best_co2": float(optimal_solutions[np.argmin(optimal_solutions[:, 1]), 1]),
            "co2_solution": optimal_solutions[np.argmin(optimal_solutions[:, 1])].tolist(),
            "best_cost": float(optimal_solutions[np.argmin(optimal_solutions[:, 2]), 2]),
            "cost_solution": optimal_solutions[np.argmin(optimal_solutions[:, 2])].tolist()
        }

        # 1) Generate multi-subplot scatter plot
        subplot_fig, axs = plt.subplots(5, 2, figsize=(15, 20))
        axs = axs.flatten()

        # Create DataFrame for the feasible solutions
        optimal_df = pd.DataFrame(optimal_features, columns=required_cols)
        optimal_df["Strength Deviation (MPa)"] = optimal_solutions[:, 0]
        optimal_df["CO2 (kg/m^3)"] = optimal_solutions[:, 1]
        optimal_df["Cost (INR/m^3)"] = optimal_solutions[:, 2]

        co2_min, co2_max = optimal_df["CO2 (kg/m^3)"].min(), optimal_df["CO2 (kg/m^3)"].max()
        co2_range = co2_max - co2_min if (co2_max - co2_min) != 0 else 1e-9
        # We'll scale the marker size by CO2
        optimal_df["CO2_normalized"] = ((optimal_df["CO2 (kg/m^3)"] - co2_min) / co2_range) * 250

        # Plot each feature vs Cost, color by Strength Dev, size by CO2
        for i, feature in enumerate(required_cols):
            sc = axs[i].scatter(
                optimal_df["Cost (INR/m^3)"],
                optimal_df[feature],
                c=optimal_df["Strength Deviation (MPa)"],
                cmap="viridis",
                s=optimal_df["CO2_normalized"],
                alpha=0.7
            )
            # Use a smaller colorbar fraction & shrink to avoid horizontal scroll
            cbar = subplot_fig.colorbar(sc, ax=axs[i], label="Strength Deviation (MPa)",
                                        fraction=0.046, pad=0.04, shrink=1)
            axs[i].set_xlabel("Cost (INR/m³)")
            axs[i].set_ylabel(feature)

        # Adjust spacing to fit color bars
        plt.tight_layout()
        # You can also tweak subplots_adjust if needed:
        subplot_fig.subplots_adjust(left=0.06, right=0.94, top=0.95, bottom=0.05)

        # Save figure to buffer
        buf = BytesIO()
        subplot_fig.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode()

        # 2) Generate an enlarged, centered 3D Plotly chart
        fig3d = go.Figure(
            data=[go.Scatter3d(
                x=optimal_solutions[:, 1].tolist(),
                y=optimal_solutions[:, 2].tolist(),
                z=optimal_solutions[:, 0].tolist(),
                mode="markers",
                marker=dict(
                    size=5,
                    color=optimal_solutions[:, 0],
                    colorscale="Viridis",
                    opacity=0.8,
                    colorbar=dict(title="Strength Dev.")
                )
            )]
        )
        fig3d.update_layout(
            scene=dict(
                xaxis_title="CO2 (kg/m³)",
                yaxis_title="Cost (INR/m³)",
                zaxis_title="Strength Deviation (MPa)",
                camera=dict(eye=dict(x=2, y=2, z=2))
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            title="⚖️ Multi-Objective Optimization: Balancing Strength, Cost & Emissions",
            width=1000,
            height=800
        )
        plotly_html = fig3d.to_html(full_html=False)

        # Build HTML tables for objectives & mix parameters
        obj_df_html = pd.DataFrame(
            optimal_solutions,
            columns=["Strength Deviation (MPa)", "CO2 (kg/m^3)", "Cost (INR/m^3)"]
        ).to_html(classes="table", index=False)

        mix_df_html = pd.DataFrame(optimal_features, columns=required_cols).to_html(classes="table", index=False)

        # 3) Build PDF (omits the multi-subplot figure)
        inputs_pdf = {
            'desired_strength': desired_strength,
            'expected_date': expected_date,
            'iterations': iterations,
            'cement_range': [min_max_values.loc["Cement", "min"], min_max_values.loc["Cement", "max"]],
            'water_range': [min_max_values.loc["Water", "min"], min_max_values.loc["Water", "max"]]
        }
        pdf_buffer = create_optimization_pdf(inputs_pdf, results, obj_df_html, mix_df_html, scaled_optimal_solutions)
        pdf_data = pdf_buffer.getvalue()
        pdf_base64 = base64.b64encode(pdf_data).decode()

        # Return the rendered page
        return render_template("mix_optimization.html",
                               form_data=request.form,
                               results=results,
                               plot_url=plot_url,
                               plotly_html=plotly_html,
                               pdf_base64=pdf_base64,
                               obj_df=obj_df_html,
                               mix_df=mix_df_html)
    else:
        # GET => Show form with default values
        return render_template("mix_optimization.html",
                               form_data=default_values,
                               results=None,
                               plot_url=None,
                               plotly_html=None,
                               pdf_base64=None,
                               obj_df="",
                               mix_df="")


###########################################
# Example helper that calculates the mix
###########################################
def calculate_mix_proportion(form_data):
    """
    Takes the form_data (dict) from request.form and returns a 'results' dict
    for display & PDF creation.
    """
    # Extract fields
    GRADE = form_data.get('GRADE', '').upper().replace('M', 'M ')
    TYPE_OF_MINERAL_ADMIXTURE = form_data.get('TYPE_OF_MINERAL_ADMIXTURE', '')
    SP_ADMIX = float(form_data.get('SP_ADMIX', 0)) if TYPE_OF_MINERAL_ADMIXTURE == "Fly ash" else None
    SIZE_OF_AGGREGATE = form_data.get('SIZE_OF_AGGREGATE', '')
    WORKABILITY = float(form_data.get('WORKABILITY', 0))
    EXPOSURE_CONDITION = form_data.get('EXPOSURE_CONDITION', '')
    TYPECONC = form_data.get('TYPECONC', '')
    # Checkboxes for pumping, moisture, etc. are "on" if checked
    pumping = ('pumping' in form_data)  # True if user checked it
    TYPE_OF_AGGREGATE = form_data.get('TYPE_OF_AGGREGATE', '')
    CHEMICAL_ADMIXTURE = form_data.get('CHEMICAL_ADMIXTURE', '')
    SP_CEMENT = float(form_data.get('SP_CEMENT', 0))
    SP_CA = float(form_data.get('SP_CA', 0))
    SP_FA = float(form_data.get('SP_FA', 0))
    SP_CHEMAD = float(form_data.get('SP_CHEMAD', 1.15))
    WATER_ABSORPTION_CA = float(form_data.get('WATER_ABSORPTION_CA', 0))
    WATER_ABSORPTION_FA = float(form_data.get('WATER_ABSORPTION_FA', 0))
    ZONE_OF_FA = form_data.get('ZONE_OF_FA', '')
    CA_SURFACE_MOISTURE = "Yes" if ('CA_SURFACE_MOISTURE' in form_data) else "No"
    FA_SURFACE_MOISTURE = "Yes" if ('FA_SURFACE_MOISTURE' in form_data) else "No"

    # Now do your actual mix calculations:
    TARGET_STRENGTH = target_strength_calculation(GRADE)
    WATER_CEMENT_RATIO = water_cement_ratio_calculation(EXPOSURE_CONDITION, TYPECONC)
    WATER_CONTENT = water_content_calculation(WORKABILITY, SIZE_OF_AGGREGATE, TYPE_OF_AGGREGATE, CHEMICAL_ADMIXTURE)

    if TYPE_OF_MINERAL_ADMIXTURE == "":
        CEMENT_CONTENT = cement_content_calculation(EXPOSURE_CONDITION, WATER_CEMENT_RATIO, WATER_CONTENT, TYPECONC)
        VOL_CA, VOL_FA = vol_of_CAnFA_calculation(ZONE_OF_FA, SIZE_OF_AGGREGATE, WATER_CEMENT_RATIO, pumping)
        MASS_CHEM_AD, MASS_CA, MASS_FA = mix_calculation(CEMENT_CONTENT, SP_CEMENT, WATER_CONTENT, VOL_CA, VOL_FA,
                                                         SP_CA, SP_FA, SP_CHEMAD)
        FLYASH_CONTENT = None
        NEW_WATER_CEMENT_RATIO = None
    else:
        CEMENT_CONTENT, FLYASH_CONTENT, _, NEW_WATER_CEMENT_RATIO = fly_cement_content_calculation(
            EXPOSURE_CONDITION, WATER_CEMENT_RATIO, WATER_CONTENT, TYPECONC
        )
        VOL_CA, VOL_FA = vol_of_CAnFA_calculation(ZONE_OF_FA, SIZE_OF_AGGREGATE, WATER_CEMENT_RATIO, pumping)
        MASS_CHEM_AD, MASS_CA, MASS_FA = fly_mix_calculation(
            CEMENT_CONTENT, SP_CEMENT, WATER_CONTENT, VOL_CA, VOL_FA,
            SP_CA, SP_FA, SP_ADMIX, SP_CHEMAD, FLYASH_CONTENT
        )

    CA_WA = MASS_CA * WATER_ABSORPTION_CA * 0.01
    FA_WA = MASS_FA * WATER_ABSORPTION_FA * 0.01
    CA_SM = MASS_CA * 0.01 if CA_SURFACE_MOISTURE == "Yes" else 0
    FA_SM = MASS_FA * 0.01 if FA_SURFACE_MOISTURE == "Yes" else 0
    FREE_WATER = WATER_CONTENT + CA_WA + FA_WA - CA_SM - FA_SM

    # Build results dict
    results = {
        "GRADE": GRADE,
        "TARGET_STRENGTH": TARGET_STRENGTH,
        "WATER_CEMENT_RATIO": WATER_CEMENT_RATIO,
        "WATER_CONTENT": WATER_CONTENT,
        "CEMENT_CONTENT": CEMENT_CONTENT,
        "MASS_CHEM_AD": MASS_CHEM_AD,
        "MASS_CA": MASS_CA,
        "MASS_FA": MASS_FA,
        "CA_WA": CA_WA,
        "FA_WA": FA_WA,
        "CA_SM": CA_SM,
        "FA_SM": FA_SM,
        "FREE_WATER": FREE_WATER,
    }
    if TYPE_OF_MINERAL_ADMIXTURE == "Fly ash":
        results["FLYASH_CONTENT"] = FLYASH_CONTENT
        results["NEW_WATER_CEMENT_RATIO"] = NEW_WATER_CEMENT_RATIO

    return results

#############################################
# Generate Mix Route (Unified: POST => HTML)
#############################################
@app.route("/generate_mix", methods=["GET", "POST"])
def generate_mix():
    if request.method == "POST":
        form_data = request.form.to_dict()
        results = calculate_mix_proportion(form_data)
        session["mix_results"] = results
        session["mix_form_data"] = form_data

        # Generate pie chart
        labels = ["Cement", "Water", "Fine Aggregate", "Coarse Aggregate", "Chemical Admixture"]
        values = [
            results.get('CEMENT_CONTENT', 0),
            results.get('WATER_CONTENT', 0),
            results.get('MASS_FA', 0),
            results.get('MASS_CA', 0),
            results.get('MASS_CHEM_AD', 0)
        ]
        if form_data.get('TYPE_OF_MINERAL_ADMIXTURE') == "Fly ash":
            labels.insert(1, "Fly Ash")
            values.insert(1, results.get('FLYASH_CONTENT', 0))

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#6a5acd', '#ff5733', '#20c997', '#a569bd', '#f4c242', '#e0e0e0']
        patches, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct=lambda pct: f'{pct:.1f}%',
            startangle=140,
            colors=colors,
            wedgeprops={'edgecolor': 'black'}
        )
        for text in autotexts:
            text.set_color('white')
            text.set_fontsize(10)
        ax.axis('equal')

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        # Encode the pie chart image to base64
        pie_chart_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return render_template(
            "generate_mix.html",
            form_data=form_data,
            results=results,
            pie_chart_base64=pie_chart_base64
        )
    else:
        default_form = {
            "GRADE": "M40",
            "TYPE_OF_MINERAL_ADMIXTURE": "",
            "SP_ADMIX": "",
            "SIZE_OF_AGGREGATE": "20",
            "WORKABILITY": "100",
            "EXPOSURE_CONDITION": "Mild",
            "TYPECONC": "Plain",
            "TYPE_OF_AGGREGATE": "Sub-angular",
            "CHEMICAL_ADMIXTURE": "Super Plasticizer",
            "SP_CEMENT": "3.15",
            "SP_CA": "2.7",
            "SP_FA": "2.65",
            "SP_CHEMAD": "1.15",
            "WATER_ABSORPTION_CA": "0.5",
            "WATER_ABSORPTION_FA": "0.5",
            "ZONE_OF_FA": "Zone 2",
        }
        return render_template("generate_mix.html", form_data=default_form, results=None, pie_chart_base64=None)

#############################################
# Download PDF Route for Mix
#############################################
@app.route("/download_mix_pdf", methods=["POST"])
def download_mix_pdf():
    results = session.get("mix_results", {})
    form_data = session.get("mix_form_data", {})

    labels = ["Cement", "Water", "Fine Aggregate", "Coarse Aggregate", "Chemical Admixture"]
    values = [
        results.get('CEMENT_CONTENT', 0),
        results.get('WATER_CONTENT', 0),
        results.get('MASS_FA', 0),
        results.get('MASS_CA', 0),
        results.get('MASS_CHEM_AD', 0)
    ]
    if form_data.get('TYPE_OF_MINERAL_ADMIXTURE') == "Fly ash":
        labels.insert(1, "Fly Ash")
        values.insert(1, results.get('FLYASH_CONTENT', 0))

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#6a5acd', '#ff5733', '#20c997', '#a569bd', '#f4c242', '#e0e0e0']
    patches, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct=lambda pct: f'{pct:.1f}%',
        startangle=140,
        colors=colors,
        wedgeprops={'edgecolor': 'black'}
    )
    for text in autotexts:
        text.set_color('white')
        text.set_fontsize(10)
    ax.axis('equal')

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    pdf_buffer = create_mix_pdf(form_data, results, buf)

    return send_file(pdf_buffer, as_attachment=True,
                     download_name='Mix_Design_Report.pdf',
                     mimetype='application/pdf')

@app.route("/strength_prediction")
def strength_prediction():
    return render_template("strength_prediction.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        cement = float(request.form['Cement'])
        blast_furnace_slag = float(request.form['BlastFurnaceSlag'])
        fly_ash = float(request.form['FlyAsh'])
        water = float(request.form['Water'])
        super_plasticizer = float(request.form['SuperPlasticizer'])
        coarse_aggregate = float(request.form['CoarseAggregate'])
        fine_aggregate = float(request.form['FineAggregate'])
        age = int(request.form['Age'])
        model_choice = request.form['model_choice']
    except Exception as e:
        return f"Error in input parameters: {e}", 400

    # Create numpy array with input features
    input_array = np.array([[cement, blast_furnace_slag, fly_ash, water,
                             super_plasticizer, coarse_aggregate, fine_aggregate, age]])

    # Scale the input using the loaded scaler (scaler_X should be defined in predict.py or in this file)
    input_scaled = scaler_X.transform(input_array)

    if model_choice == "ANN":
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_scaled = model_ann(input_tensor).cpu().numpy()
        pred_orig = scaler_y.inverse_transform(pred_scaled)[0, 0]
        result = f"[ANN] Predicted Concrete Strength: {pred_orig:.2f} MPa"
    elif model_choice == "XGBoost":
        pred_norm = xgb_model.predict(input_scaled)
        pred_orig = scaler_y.inverse_transform(pred_norm.reshape(-1, 1))[0, 0]
        result = f"[XGBoost] Predicted Concrete Strength: {pred_orig:.2f} MPa"
    else:
        result = "Invalid model choice."

    # Render the strength_prediction template with the result
    return render_template("strength_prediction.html", prediction_result=result)

    

if __name__ == "__main__":
    app.run(debug=True, port=8000)