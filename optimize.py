import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from platypus import NSGAII, Problem, Real
import joblib
from io import BytesIO
from fpdf import FPDF

def create_optimization_pdf(inputs, results, objectives_df, mix_df, scaled_obj_df):
    # Dummy PDF for optimization
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'Optimization Report', 0, 1, 'C')
    pdf.set_font('Helvetica', '', 12)
    pdf.multi_cell(0, 10, "This is a dummy optimization report generated for demonstration purposes.")
    pdf_bytes = pdf.output(dest='S')
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode('latin1')
    return BytesIO(pdf_bytes)

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
            model_file = os.path.join(path, f"XGB_{name}_model.pkl")
            models[name] = joblib.load(model_file)
        return models

    def optimize_concrete_mix(self, x):
        cement, clinker, slag, flyash, limestone, gypsum, water, super_plasticizer, coarse_agg, fine_agg = x
        input_features = [
            cement, clinker, slag, flyash, limestone,
            gypsum, water, super_plasticizer, coarse_agg, fine_agg,
            self.expected_date
        ]
        strength_pred = self.models["Strength"].predict([input_features])[0]
        co2_pred = self.models["CO2"].predict([input_features])[0]
        cost_pred = self.models["Cost"].predict([input_features])[0]
        obj_strength_dev = abs(strength_pred - self.desired_strength)
        objectives = [obj_strength_dev, co2_pred, cost_pred]
        constraints = [self.desired_strength - strength_pred]
        return objectives, constraints

    def run_optimization(self, iterations):
        problem = Problem(10, 3, 1)
        problem.types[:] = [
            Real(self.min_max_values.loc[feat, "min"], self.min_max_values.loc[feat, "max"])
            for feat in self.feature_columns
        ]
        problem.function = self.optimize_concrete_mix
        problem.constraints[:] = "<=0"
        algorithm = NSGAII(problem)
        algorithm.run(iterations)
        feasible_solutions = [s for s in algorithm.result if s.feasible]
        optimal_solutions = np.array([[s.objectives[0], s.objectives[1], s.objectives[2]] for s in feasible_solutions])
        optimal_features = np.array([s.variables for s in feasible_solutions])
        if optimal_solutions.size > 0:
            scaler = MinMaxScaler()
            scaled_optimal_solutions = scaler.fit_transform(optimal_solutions)
        else:
            scaled_optimal_solutions = optimal_solutions
        return optimal_solutions, scaled_optimal_solutions, optimal_features

def optimize_mix_logic(data):
    # Convert inputs
    cement = float(data.get("cement", 0))
    clinker = float(data.get("clinker", 0))
    slag = float(data.get("slag", 0))
    flyash = float(data.get("flyash", 0))
    limestone = float(data.get("limestone", 0))
    gypsum = float(data.get("gypsum", 0))
    water = float(data.get("water", 0))
    super_plasticizer = float(data.get("super_plasticizer", 0))
    coarse_agg = float(data.get("coarse_agg", 0))
    fine_agg = float(data.get("fine_agg", 0))
    desired_strength = float(data.get("desired_strength", 30))
    expected_date = float(data.get("expected_date", 28))
    iterations = int(data.get("iterations", 1000))

    # Load bounds from CSV or fallback
    try:
        df = pd.read_csv("OptData.csv")
        df.columns = [col.strip() for col in df.columns]
        if "Age (days)" in df.columns:
            df = df.drop(columns=["Age (days)", "Strength (MPa)", "CO2 (kg/m^3)", "Cost (INR/m^3)"])
        min_max_values = df.describe().loc[["min", "max"]].transpose()
    except Exception:
        # fallback
        min_max_values = pd.DataFrame({
            "min": [200,150,0,0,0,0,100,0,800,700],
            "max": [510,300,100,100,50,20,250,10,1200,1000]
        }, index=["Cement","Clinker","Slag","FlyAsh","Limestone","Gypsum","Water","SuperPlasticizer","CoarseAggregate","FineAggregate"])

    feature_columns = min_max_values.index

    # Run optimization
    optimizer = ConcreteOptimizer("./Mod", min_max_values, feature_columns, desired_strength, expected_date)
    optimal_solutions, scaled_optimal_solutions, optimal_features = optimizer.run_optimization(iterations)

    if optimal_solutions.size > 0:
        # For the 3D plot, let's convert all solutions into arrays we can send back
        all_solutions = []
        for sol in optimal_solutions:
            all_solutions.append({
                "strength_dev": float(sol[0]),
                "co2": float(sol[1]),
                "cost": float(sol[2])
            })
        # Return first feasible solution
        best_sol = optimal_solutions[0]
        best_feats = optimal_features[0]
        obj = {"strength_dev": float(best_sol[0]),
               "co2": float(best_sol[1]),
               "cost": float(best_sol[2])}
        mix_params = {
            "cement": float(best_feats[0]),
            "clinker": float(best_feats[1]),
            "slag": float(best_feats[2]),
            "flyash": float(best_feats[3]),
            "limestone": float(best_feats[4]),
            "gypsum": float(best_feats[5]),
            "water": float(best_feats[6]),
            "super_plasticizer": float(best_feats[7]),
            "coarse_agg": float(best_feats[8]),
            "fine_agg": float(best_feats[9])
        }
        response = {
            "success": True,
            "message": "Optimization completed successfully.",
            # We'll pass back all solutions so we can do a 3D Plotly scatter
            "allSolutions": all_solutions,
            "objectives": [obj],
            "mixParams": [mix_params],
            "pdfData": {"someKey": "someValue"}
        }
    else:
        response = {
            "success": False,
            "message": "No feasible solution found. Please adjust your parameters and try again."
        }
    return response
