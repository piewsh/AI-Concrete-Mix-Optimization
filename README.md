# 🧱 MixXperts — AI-Driven Concrete Mix Design & Optimization

> **Multi-objective concrete mix optimization using GNN, ANN, XGBoost, and NSGA-II with IS Code compliance**
> *Winner — TECH TONIC Software Hackathon, NIT Puducherry*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-Web_UI-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com)

---

## 📋 What This Does

MixXperts is a full-stack web application that combines **machine learning** with **Indian Standard (IS) code calculations** to design and optimize concrete mix proportions. It targets three objectives simultaneously: **compressive strength**, **cost**, and **CO₂ emissions**.

### 🏆 Impact
- Reduces trial batches by up to **60%**
- Generates IS 456:2000 and IS 10262:2009 compliant mix designs
- Produces downloadable PDF reports for site engineers

---

## 🧠 Models

### Graph Neural Network (ConcreteGNN)
A custom **message-passing GNN** built with PyTorch Geometric that models concrete components as graph nodes with inter-material interactions:
- `Encoder` → `ProcessorBlock` (×6 MetaLayer passes) → `Decoder`
- Edge, Node, and Global update models with residual connections
- Captures non-linear relationships between cement, SCMs, aggregates, and water

### Artificial Neural Network (ConcreteANN)
- PyTorch feedforward network trained on standardized concrete datasets
- StandardScaler preprocessing, ReduceLROnPlateau scheduler
- Validated with R² score on held-out test set

### XGBoost Ensemble
- Separate XGBoost models for **Strength**, **CO₂**, and **Cost** prediction
- Random Forest models as secondary baselines
- Pre-trained `.pkl` model files included

### NSGA-II Multi-Objective Optimization
- **Platypus** framework for Pareto-optimal mix design
- 10 decision variables: Cement, Clinker, Slag, Fly Ash, Limestone, Gypsum, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate
- 3 objectives: Minimize strength deviation, CO₂ emissions, and cost
- Constraint: Predicted strength ≥ desired strength

---

## 🏗️ IS Code Integration (`concMixIS.py`)

Implements concrete mix design calculations per Indian Standards:
- **IS 456:2000** — Exposure-based minimum cement content & max w/c ratio
- **IS 10262:2009** — Target strength, water content, aggregate volume calculations
- Fly ash substitution calculations included

---

## 🖥️ Web Application

Interactive Flask dashboard with 4 modules:
1. **Strength Prediction** — Input mix proportions → predict compressive strength
2. **IS Code Mix Design** — Generate compliant mix proportions from grade & exposure
3. **Multi-Objective Optimization** — NSGA-II Pareto front with interactive Plotly 3D scatter
4. **PDF Report Generation** — Downloadable mix design reports

---

## 📁 Project Structure

```
MixXperts/
├── app.py                    # Flask web server (533 lines)
├── model.py                  # ConcreteGNN (PyTorch Geometric)
├── trainann.py               # ConcreteANN definition
├── trainannModel.py          # ANN training pipeline
├── train.py                  # GNN training pipeline
├── train_xgb.py              # XGBoost training
├── predict.py                # Prediction utilities
├── optimize.py               # NSGA-II optimizer (Platypus)
├── concMixIS.py              # IS 456/10262 code calculations
├── pdfMaker.py               # PDF report generator
├── concrete.csv              # Training dataset
├── OptData.csv               # Optimization dataset
├── ann_model.pth             # Trained ANN weights
├── Mod/                      # Pre-trained XGBoost & RF models
├── templates/                # Flask HTML templates
├── static/                   # CSS, JS, assets
└── requirements.txt
```

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## 👤 Author

**Piyush Ranjan Singh** — [GitHub](https://github.com/piewsh) • [Email](mailto:rajputpiyush2009@gmail.com)
