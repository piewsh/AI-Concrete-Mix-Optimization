import os
import base64
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from flask import render_template, request
from io import BytesIO
from datetime import datetime
import pytz
from sklearn.preprocessing import MinMaxScaler
from platypus import NSGAII, Problem, Real
from fpdf import FPDF
import tempfile
from tempfile import NamedTemporaryFile

class PDF(FPDF):
    def header(self):
        try:
            self.image("static/1.png", x=10, y=8, w=30)
        except Exception as e:
            print(f"Header image error: {e}")
        self.set_xy(150, 15)
        self.set_font('Arial', 'B', 10)
        self.cell(40, 5, "Project Details:", 0, 2)
        self.set_font('Arial', '', 9)
        self.cell(40, 5, "Mix Design Report", 0, 2)
        self.cell(40, 5, "Version: 1.0", 0, 2)
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist)
        self.cell(40, 5, f"Date: {now_ist.strftime('%Y-%m-%d %H:%M:%S %Z')}", 0, 2)
        self.line(10, 40, 200, 40)
        self.set_y(50)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Downloaded from MixXperts', 0, 0, 'C')

def create_mix_pdf(inputs, results, image_buffer):
    pdf = PDF()
    pdf.add_page()

    # Title Section
    pdf.set_y(50)
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Mix Design Report', 0, 1, 'C')
    pdf.set_font('Arial', 'I', 12)
    pdf.cell(0, 10, '(as per IS 10262)', 0, 1, 'C')
    pdf.ln(10)
    
    # Input Parameters Section
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Input Parameters:', 0, 1)
    pdf.set_font('Arial', '', 12)
    inputs_text = [
        f"Grade Designation: {inputs.get('GRADE','')}",
        f"Mineral Admixture: {inputs.get('TYPE_OF_MINERAL_ADMIXTURE','None') or 'None'}",
        f"Aggregate Size: {inputs.get('SIZE_OF_AGGREGATE','')} mm",
        f"Workability (Slump): {inputs.get('WORKABILITY','')} mm",
        f"Exposure Condition: {inputs.get('EXPOSURE_CONDITION','')}",
        f"Concrete Type: {inputs.get('TYPECONC','')}",
        f"Pumping: {'Yes' if inputs.get('pumping', False) else 'No'}",
        f"Aggregate Type: {inputs.get('TYPE_OF_AGGREGATE','')}",
        f"Chemical Admixture: {inputs.get('CHEMICAL_ADMIXTURE','')}",
        f"Specific Gravity - Cement: {inputs.get('SP_CEMENT','')}",
        f"Specific Gravity - Coarse Aggregate: {inputs.get('SP_CA','')}",
        f"Specific Gravity - Fine Aggregate: {inputs.get('SP_FA','')}",
        f"Specific Gravity - Chemical Admixture: {inputs.get('SP_CHEMAD','')}",
        f"Water Absorption - Coarse Aggregate: {inputs.get('WATER_ABSORPTION_CA','')}%",
        f"Water Absorption - Fine Aggregate: {inputs.get('WATER_ABSORPTION_FA','')}%",
        f"Fine Aggregate Zone: {inputs.get('ZONE_OF_FA','')}",
        f"Surface Moisture - Coarse Aggregate: {inputs.get('CA_SURFACE_MOISTURE','')}",
        f"Surface Moisture - Fine Aggregate: {inputs.get('FA_SURFACE_MOISTURE','')}"
    ]
    for line in inputs_text:
        pdf.cell(0, 10, line, 0, 1)
    pdf.ln(5)
    
    # Calculations Section
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 15, 'Calculations:', 0, 1)
    pdf.set_font('Arial', '', 12)
    calc_text = [
        f"Target Strength: {results.get('TARGET_STRENGTH', 0):.2f} N/mm²",
        f"Water-Cement Ratio: {results.get('WATER_CEMENT_RATIO', 0):.3f}",
        f"Water Content: {results.get('WATER_CONTENT', 0):.2f} lit/m³",
        f"Cement Content: {results.get('CEMENT_CONTENT', 0):.2f} kg/m³"
    ]
    if inputs.get('TYPE_OF_MINERAL_ADMIXTURE','') == "Fly ash":
        calc_text.append(f"Fly Ash Content: {results.get('FLYASH_CONTENT', 0):.2f} kg/m³")
        calc_text.append(f"Adjusted Water-Cement Ratio: {results.get('NEW_WATER_CEMENT_RATIO', 0):.3f}")
    for line in calc_text:
        pdf.cell(0, 10, line, 0, 1)
    pdf.ln(5)
    
    # Mix Proportions Section
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Mix Proportions:', 0, 1)
    pdf.set_font('Arial', '', 12)
    mix_text = [
        f"Cement: {results.get('CEMENT_CONTENT', 0):.2f} kg/m³",
        f"Water: {results.get('WATER_CONTENT', 0):.2f} lit/m³",
        f"Fine Aggregate: {results.get('MASS_FA', 0):.2f} kg/m³",
        f"Coarse Aggregate: {results.get('MASS_CA', 0):.2f} kg/m³",
        f"Chemical Admixture: {results.get('MASS_CHEM_AD', 0):.2f} kg/m³"
    ]
    if inputs.get('TYPE_OF_MINERAL_ADMIXTURE','') == "Fly ash":
        mix_text.insert(1, f"Fly Ash: {results.get('FLYASH_CONTENT', 0):.2f} kg/m³")
    for line in mix_text:
        pdf.cell(0, 10, line, 0, 1)
    pdf.ln(5)
    
    # Corrections Section
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Corrections:', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"Coarse Aggregate Water Absorption: {results.get('CA_WA', 0):.2f} lit", 0, 1)
    pdf.cell(0, 10, f"Fine Aggregate Water Absorption: {results.get('FA_WA', 0):.2f} lit", 0, 1)
    if inputs.get('CA_SURFACE_MOISTURE','') == "Yes" or inputs.get('FA_SURFACE_MOISTURE','') == "Yes":
        pdf.cell(0, 10, f"Coarse Aggregate Surface Moisture: {results.get('CA_SM', 0):.2f} lit", 0, 1)
        pdf.cell(0, 10, f"Fine Aggregate Surface Moisture: {results.get('FA_SM', 0):.2f} lit", 0, 1)
    pdf.cell(0, 10, f"Final Free Water Content: {results.get('FREE_WATER', 0):.2f} lit", 0, 1)
    pdf.ln(10)
    
    # Pie Chart on New Page
    with NamedTemporaryFile(delete=False, suffix='.png') as tmp_img:
        tmp_img.write(image_buffer.getvalue())
        tmp_img_path = tmp_img.name
    pdf.add_page()
    pdf.image(tmp_img_path, x=10, y=50, w=190)
    os.unlink(tmp_img_path)
    
    pdf_bytes = pdf.output(dest='S')
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode('latin1')
    pdf_buffer = BytesIO(pdf_bytes)
    pdf_buffer.seek(0)
    return pdf_buffer


#############################################
# Optimization PDF Generation Class
#############################################
class PDF2(FPDF):
    """
    This PDF class omits the multi-subplot scatter plot,
    so it only shows input parameters, a summary table, and a concluding note.
    """
    def __init__(self, orientation='P', unit='mm', format='A4'):
        super().__init__(orientation=orientation, unit=unit, format=format)
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        try:
            self.image("static/1.png", x=10, y=8, w=30)
        except Exception as e:
            print(f"Header image error: {e}")
        self.set_xy(150, 15)
        self.set_font('Helvetica', 'B', 10)
        self.cell(40, 5, "Project Details:", 0, 2)
        self.set_font('Helvetica', '', 9)
        self.cell(40, 5, "Mix Design Report", 0, 2)
        self.cell(40, 5, "Version: 1.0", 0, 2)
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist)
        self.cell(40, 5, f"Date: {now_ist.strftime('%Y-%m-%d %H:%M:%S %Z')}", 0, 2)
        self.line(10, 40, 200, 40)
        self.set_y(50)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, 'Downloaded from MixXperts', 0, 0, 'C')

    def optimization_report(self, inputs, results, obj_df_html, mix_df_html, scaled_obj_df):
        self.add_page()
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, 'Concrete Mix Optimization Report', 0, 1, 'C')
        self.ln(10)
        # Input Parameters Section
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'Optimization Parameters:', 0, 1)
        self.set_font('Helvetica', '', 11)
        for line in [
            f"Desired Strength: {inputs['desired_strength']} MPa",
            f"Testing Age: {inputs['expected_date']} days",
            f"Iterations: {inputs['iterations']}",
            f"Cement Range: {inputs['cement_range'][0]} - {inputs['cement_range'][1]} kg/m³",
            f"Water Range: {inputs['water_range'][0]} - {inputs['water_range'][1]} kg/m³"
        ]:
            self.cell(0, 8, line, 0, 1)
        self.ln(5)
        # Optimal Solutions Summary Section
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'Optimal Solutions Summary:', 0, 1)
        self.set_font('Helvetica', '', 11)
        headers = ["Parameter", "Best Strength", "Lowest CO2", "Lowest Cost"]
        data = [
            ["Strength Dev (MPa)",
             f"{results['best_strength_dev']:.2f}",
             f"{results['co2_solution'][0]:.2f}",
             f"{results['cost_solution'][0]:.2f}"],
            ["CO2 (kg/m³)",
             f"{results['strength_solution'][1]:.2f}",
             f"{results['best_co2']:.2f}",
             f"{results['cost_solution'][1]:.2f}"],
            ["Cost (INR/m³)",
             f"{results['strength_solution'][2]:.2f}",
             f"{results['co2_solution'][2]:.2f}",
             f"{results['best_cost']:.2f}"]
        ]
        col_widths = [50, 45, 45, 45]
        row_height = 10
        for i, col in enumerate(headers):
            self.cell(col_widths[i], row_height, col, 1, 0, 'C')
        self.ln()
        for row in data:
            for i, item in enumerate(row):
                self.cell(col_widths[i], row_height, item, 1, 0, 'C')
            self.ln()
        self.ln(10)
        # Conclusion Section
        self.set_font('Helvetica', 'I', 11)
        self.multi_cell(0, 8,
            "Note: The optimal solutions represent trade-offs between competing objectives. "
            "Final selection should consider project-specific requirements and material availability."
        )

def create_optimization_pdf(inputs, results, obj_df_html, mix_df_html, scaled_obj_df):
    pdf = PDF2()
    pdf.optimization_report(inputs, results, obj_df_html, mix_df_html, scaled_obj_df)
    pdf_bytes = pdf.output(dest='S')
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode('latin1')
    return BytesIO(pdf_bytes)
