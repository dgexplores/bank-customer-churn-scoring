# Bank Customer Churn Predictive Risk-Scoring System

This project implements a complete, end-to-end Machine Learning pipeline and Streamlit dashboard to predict bank customer churn, optimize retention decision thresholds, and explain predictive risk drivers.

## Project Structure

```
churn_project/
├── data/
│   └── European_Bank.csv   # Customer dataset (10,000 rows)
├── churn_utils.py          # Shared utility: column mapping and feature engineering
├── train_pipeline.py       # ML Pipeline: train, CV, evaluate, explain, and save artifacts
├── app.py                  # Streamlit Interactive Dashboard
├── requirements.txt        # Project package dependencies
├── README.md               # Setup and execution instructions
└── outputs/
    ├── figures/            # Evaluation and explainability plots (PNG)
    ├── metrics/            # CSV/JSON performance summaries
    └── models/             # Pickled scikit-learn pipeline and tuned threshold
```

## Setup Instructions

1. **Prerequisites**: Python 3.9+ installed on your system.
2. **Navigate** to the project directory:
   ```bash
   cd churn_project
   ```
3. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   ```
4. **Activate the virtual environment**:
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
5. **Upgrade pip and install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

*Note: On macOS, if you wish to run optional XGBoost model training, you may need to install `libomp` via Homebrew (`brew install libomp`) to enable OpenMP.*

## Running the Training Pipeline

The training pipeline loads the customer data, applies feature engineering, evaluates 5 distinct ML estimators (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost if available) using 5-fold Stratified Cross-Validation, optimizes the classification threshold, and generates explainability plots.

To execute the pipeline:
```bash
python train_pipeline.py
```

This will populate the `outputs/` folder with models, metrics, and figures.

## Launching the Streamlit Dashboard

To launch the interactive dashboard:
```bash
streamlit run app.py
```

Once running, open your web browser and navigate to the local address displayed in the terminal (typically `http://localhost:8501`).

### Dashboard Modules
- **Customer Churn Risk Calculator**: Predicts the probability of churn for individual customer profiles, assigning low/moderate/elevated/high risk ratings and suggesting targeted actions.
- **Probability Distribution Visualization**: Displays a histogram of predictions across the entire portfolio, marks the retention threshold, and details segment breakdowns.
- **Feature Importance Dashboard**: Evaluates global feature importances and shows SHAP summary plots.
- **What-If Scenario Simulator**: Simulates operational interventions (changing product counts and activity membership) to visualize the impact on customer risk scores.

## Swapping in a New Dataset

To score a new customer dataset:
1. Ensure the new file is formatted with similar column headers (header mapping handles minor variations in capitalization and spacing).
2. Place the new file in `data/European_Bank.csv`.
3. Re-run `python train_pipeline.py` to retrain the models and tune the threshold on the new data, then launch the app.
