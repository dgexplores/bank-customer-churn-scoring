import sys
import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure churn_project is in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import churn_utils and map it to prevent unpickling errors on Streamlit Cloud
try:
    import churn_utils
    sys.modules['churn_project.churn_utils'] = churn_utils
    sys.modules['churn_utils'] = churn_utils
except ImportError:
    pass

from churn_utils import clean_and_normalize_columns, ChurnFeatureEngineer, DROP_COLS, TARGET_COL


# Page configuration
st.set_page_config(
    page_title="Bank Customer Churn Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (clean, light, modern)
st.markdown("""
<style>
    /* Main layout improvements */
    .reportview-container {
        background: #f8f9fa;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Metrics panel styling */
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a2530;
        margin-bottom: 0.25rem;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Risk Band badges */
    .risk-badge {
        font-size: 1.25rem;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        display: inline-block;
        margin-top: 0.5rem;
        color: white;
        text-align: center;
    }
    .risk-low { background-color: #2ec4b6; }
    .risk-moderate { background-color: #ff9f1c; }
    .risk-elevated { background-color: #e71d36; }
    .risk-high { background-color: #7209b7; }
    
    /* Verdict boxes */
    .verdict-box {
        border-radius: 8px;
        padding: 1.25rem;
        margin-top: 1rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-left: 5px solid;
    }
    .verdict-flagged {
        background-color: #fff0f2;
        color: #c01e38;
        border-color: #e71d36;
    }
    .verdict-safe {
        background-color: #eefdfa;
        color: #0d887a;
        border-color: #2ec4b6;
    }
    
    /* Sidebar header */
    .sidebar-header {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        color: #1a2530;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
    }
    
    /* Segment titles */
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a2530;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #dee2e6;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load pipeline, threshold, and insights
@st.cache_resource
def load_assets():
    model_path = os.path.join(BASE_DIR, 'outputs/models/best_model.pkl')
    thresh_path = os.path.join(BASE_DIR, 'outputs/models/threshold.json')
    insights_path = os.path.join(BASE_DIR, 'outputs/metrics/insights.json')
    
    if not os.path.exists(model_path) or not os.path.exists(thresh_path) or not os.path.exists(insights_path):
        st.error("Missing required ML model pipeline, threshold, or insights files. Please run `train_pipeline.py` first.")
        st.stop()
        
    pipeline = joblib.load(model_path)
    
    with open(thresh_path, 'r') as f:
        threshold = json.load(f)['threshold']
        
    with open(insights_path, 'r') as f:
        insights = json.load(f)
        
    return pipeline, threshold, insights

# Load full portfolio data for visual distribution
@st.cache_data
def load_portfolio_data(_pipeline, threshold):
    data_path = os.path.join(BASE_DIR, 'data/European_Bank.csv')
    if not os.path.exists(data_path):
        return None
        
    df_raw = pd.read_csv(data_path)
    df = clean_and_normalize_columns(df_raw)
    
    # Exclude Exited/drops to form pipeline input
    X = df.drop(columns=[TARGET_COL] + DROP_COLS, errors='ignore')
    probs = _pipeline.predict_proba(X)[:, 1]
    
    df_out = df.copy()
    df_out['Predicted_Risk'] = probs
    df_out['Flagged'] = (probs >= threshold).astype(int)
    return df_out

# Get assets
pipeline, threshold, insights = load_assets()
portfolio_df = load_portfolio_data(pipeline, threshold)

# Sidebar layout
st.sidebar.markdown("<div class='sidebar-header'>🏦 Risk Scoring Engine</div>", unsafe_allow_html=True)
st.sidebar.markdown(f"**Best Model**: `{insights['best_model']}`")
st.sidebar.markdown(f"**Tuned Churn Threshold**: `{threshold:.4f}`")

# Sidebar summary stats
st.sidebar.markdown("---")
st.sidebar.markdown("### Portfolio At-A-Glance")
st.sidebar.write(f"- Total Customers: **{insights['n_customers']:,}**")
st.sidebar.write(f"- Historic Churn Rate: **{insights['churn_rate']:.2%}**")

st.title("🏦 Bank Customer Churn Predictive Risk Dashboard")
st.markdown("A premium, ML-powered analytics platform for customer retention scoring, risk segment mapping, and interactive what-if simulation.")

# Tabs configuration
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Customer Risk Calculator",
    "📊 Portfolio Risk Distribution",
    "🔑 Feature Importance & SHAP",
    "🔮 What-If Scenario Simulator"
])

# ----------------- TAB 1: CUSTOMER RISK CALCULATOR -----------------
with tab1:
    st.markdown("<div class='section-title'>🔍 Customer Churn Risk Calculator</div>", unsafe_allow_html=True)
    st.markdown("Specify a customer's demographic and financial profile to score their likelihood of leaving the bank.")
    
    # Input layouts (2 columns)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics & Tenure")
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"], index=0)
        gender = st.selectbox("Gender", ["Female", "Male"], index=0)
        age = st.slider("Age (Years)", min_value=18, max_value=100, value=38)
        tenure = st.slider("Tenure with Bank (Years)", min_value=0, max_value=10, value=5)
        
    with col2:
        st.subheader("Financial Profile & Activity")
        credit_score = st.slider("Credit Score", min_value=350, max_value=850, value=650)
        balance = st.number_input("Account Balance (€)", min_value=0.0, max_value=300000.0, value=50000.0, step=1000.0)
        estimated_salary = st.number_input("Estimated Annual Salary (€)", min_value=0.0, max_value=300000.0, value=100000.0, step=1000.0)
        num_of_products = st.slider("Number of Bank Products Used", min_value=1, max_value=4, value=1)
        has_credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"], index=0)
        is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"], index=0)
        
    # Convert inputs to standard format
    has_crcard_val = 1 if has_credit_card == "Yes" else 0
    is_active_val = 1 if is_active_member == "Yes" else 0
    
    # Create input DataFrame (raw features matching those fed to the pipeline)
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_crcard_val],
        'IsActiveMember': [is_active_val],
        'EstimatedSalary': [estimated_salary]
    })
    
    # Predict button
    if st.button("Score Customer Risk Profile", type="primary"):
        # Run prediction
        risk_probability = pipeline.predict_proba(input_data)[:, 1][0]
        
        # Display results in columns
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.markdown("### Risk Analysis Summary")
            st.write("Calculated Risk Probability:")
            st.markdown(f"<div style='font-size: 3rem; font-weight: 800; color: #1a2530;'>{risk_probability:.1%}</div>", unsafe_allow_html=True)
            
            # Risk band determination relative to tuned threshold
            # Threshold represents the "Elevated" trigger
            if risk_probability < 0.5 * threshold:
                risk_band = "Low"
                badge_class = "risk-low"
            elif risk_probability < threshold:
                risk_band = "Moderate"
                badge_class = "risk-moderate"
            elif risk_probability < 1.5 * threshold:
                risk_band = "Elevated"
                badge_class = "risk-elevated"
            else:
                risk_band = "High"
                badge_class = "risk-high"
                
            st.markdown(f"Risk Rating: <span class='risk-badge {badge_class}'>{risk_band} Risk</span>", unsafe_allow_html=True)
            
            # Verdict Alert
            if risk_probability >= threshold:
                st.markdown(f"""
                <div class='verdict-box verdict-flagged'>
                    ⚠️ RETENTION ACTION REQUIRED<br>
                    <span style='font-size: 0.9rem; font-weight: normal;'>
                        This customer's risk score ({risk_probability:.1%}) exceeds the tuned retention trigger threshold ({threshold:.1%}).
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='verdict-box verdict-safe'>
                    ✅ NO IMMEDIATE RETENTION ACTION REQUIRED<br>
                    <span style='font-size: 0.9rem; font-weight: normal;'>
                        This customer's risk score ({risk_probability:.1%}) is below the tuned retention trigger threshold ({threshold:.1%}).
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
        with res_col2:
            st.markdown("### 🛠️ Actionable Retention Interventions")
            st.write("Based on the scored features, the following customer interventions are recommended:")
            
            tips_delivered = 0
            
            if is_active_val == 0:
                st.markdown("""
                * **Active Membership Drive**:
                  Offer fee-waivers or cashback rewards contingent on setting up 3 auto-debits or logging in to the mobile banking app weekly.
                """)
                tips_delivered += 1
                
            if num_of_products >= 3:
                st.markdown("""
                * **Product Complexity Relief**:
                  Customers with 3+ products show very high churn rates. Coordinate a customer-success follow-up call to simplify their accounts or resolve customer support tickets.
                """)
                tips_delivered += 1
                
            if geography == "Germany":
                st.markdown("""
                * **German Region Loyalty Plan**:
                  German accounts experience high churn. Offer dedicated regional savings rates, German language helpline access, or personalized local perks.
                """)
                tips_delivered += 1
                
            if balance == 0:
                st.markdown("""
                * **Deposit Booster Promotion**:
                  Send a target deposit incentive (e.g. deposit €10,000 for a €100 bonus) to secure core balance engagement.
                """)
                tips_delivered += 1
                
            if age > 40 and num_of_products == 1:
                st.markdown("""
                * **Cross-Sell Stickiness**:
                  Older single-product clients are prime targets for churn. Offer cross-selling options like retirement planners, mutual funds, or credit cards.
                """)
                tips_delivered += 1
                
            if tips_delivered < 2:
                st.markdown("""
                * **Standard Loyalty Incentives**:
                  Offer a free credit-card annual fee waiver or check-in on their experience with the bank's digital portals.
                * **Financial Relationship Review**:
                  Initiate a comprehensive financial checkup review to ensure their products align with current needs.
                """)

# ----------------- TAB 2: PORTFOLIO RISK DISTRIBUTION -----------------
with tab2:
    st.markdown("<div class='section-title'>📊 Portfolio Risk Distribution</div>", unsafe_allow_html=True)
    st.markdown("Aggregate overview of risk scoring distributions across the entire customer portfolio.")
    
    if portfolio_df is not None:
        # Portfolio Metrics Row
        tot_custs = len(portfolio_df)
        avg_risk = portfolio_df['Predicted_Risk'].mean()
        tot_flagged = portfolio_df['Flagged'].sum()
        flagged_rate = portfolio_df['Flagged'].mean()
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{tot_custs:,}</div>
                <div class='metric-label'>Total Scored Portfolio</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{avg_risk:.2%}</div>
                <div class='metric-label'>Average Churn Risk</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{tot_flagged:,}</div>
                <div class='metric-label'>Retention Flagged Customers</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_col4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{flagged_rate:.2%}</div>
                <div class='metric-label'>Portfolio Flagged Rate</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Distribution plot and analysis
        dist_col1, dist_col2 = st.columns([3, 2])
        
        with dist_col1:
            st.subheader("Predicted Churn Probability Distribution")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(portfolio_df['Predicted_Risk'], bins=50, kde=True, ax=ax, color='dodgerblue')
            ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Retention Threshold ({threshold:.3f})')
            ax.set_title("Density of Predicted Customer Churn Probabilities", fontsize=12)
            ax.set_xlabel("Predicted Churn Probability")
            ax.set_ylabel("Customer Count")
            ax.legend()
            st.pyplot(fig)
            plt.close()
            
        with dist_col2:
            st.subheader("Key Portfolio Risk Insights")
            st.write(f"""
            1. **Retention Flags**: Applying our tuned decision threshold of **{threshold:.3f}** classifies **{tot_flagged:,}** customers as high-risk. This is **{flagged_rate:.1%}** of the bank's active base.
            2. **Balanced Recall**: Rather than targeting only 20% of customers blindly, we are utilizing the custom model threshold to capture the majority of actual churners (F1 optimized), focusing budgets where it yields the highest ROI.
            3. **Primary Segment Drivers**:
               - Active members show significantly reduced churn rates.
               - German customers are highly concentrated in the high-risk region.
               - Multi-product customers (3 or 4 products) represent an immediate threat of churn.
            """)
            
        # Segment breakdowns
        st.markdown("<div class='section-title'>Segmented Churn Rate Visualizations</div>", unsafe_allow_html=True)
        seg_col1, seg_col2, seg_col3, seg_col4 = st.columns(4)
        
        # Churn by Geography
        with seg_col1:
            st.subheader("Geography Churn")
            fig, ax = plt.subplots(figsize=(4, 3))
            geog_rates = portfolio_df.groupby('Geography')['Exited'].mean()
            sns.barplot(x=geog_rates.index, y=geog_rates.values, ax=ax, palette='Blues_r')
            ax.set_ylabel("Churn Rate")
            ax.set_ylim(0, 0.45)
            for i, v in enumerate(geog_rates.values):
                ax.text(i, v + 0.01, f"{v:.1%}", ha='center', fontweight='bold')
            st.pyplot(fig)
            plt.close()
            
        # Churn by Gender
        with seg_col2:
            st.subheader("Gender Churn")
            fig, ax = plt.subplots(figsize=(4, 3))
            gender_rates = portfolio_df.groupby('Gender')['Exited'].mean()
            sns.barplot(x=gender_rates.index, y=gender_rates.values, ax=ax, palette='Purples_r')
            ax.set_ylabel("Churn Rate")
            ax.set_ylim(0, 0.45)
            for i, v in enumerate(gender_rates.values):
                ax.text(i, v + 0.01, f"{v:.1%}", ha='center', fontweight='bold')
            st.pyplot(fig)
            plt.close()
            
        # Churn by NumOfProducts
        with seg_col3:
            st.subheader("Products Churn")
            fig, ax = plt.subplots(figsize=(4, 3))
            prod_rates = portfolio_df.groupby('NumOfProducts')['Exited'].mean()
            sns.barplot(x=prod_rates.index, y=prod_rates.values, ax=ax, palette='Reds_r')
            ax.set_ylabel("Churn Rate")
            ax.set_ylim(0, 1.0)
            for i, v in enumerate(prod_rates.values):
                ax.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')
            st.pyplot(fig)
            plt.close()
            
        # Churn by Active Status
        with seg_col4:
            st.subheader("Activity Churn")
            fig, ax = plt.subplots(figsize=(4, 3))
            act_rates = portfolio_df.groupby('IsActiveMember')['Exited'].mean()
            sns.barplot(x=['Inactive', 'Active'], y=act_rates.values, ax=ax, palette='Greens_r')
            ax.set_ylabel("Churn Rate")
            ax.set_ylim(0, 0.45)
            for i, v in enumerate(act_rates.values):
                ax.text(i, v + 0.01, f"{v:.1%}", ha='center', fontweight='bold')
            st.pyplot(fig)
            plt.close()
    else:
        st.warning("Could not load full portfolio CSV data.")

# ----------------- TAB 3: FEATURE IMPORTANCE & SHAP -----------------
with tab3:
    st.markdown("<div class='section-title'>🔑 Feature Importance Dashboard</div>", unsafe_allow_html=True)
    st.markdown("Understand what features drive the churn decisions globally (based on Gini Importance) and locally (SHAP Explanations).")
    
    col_imp1, col_imp2 = st.columns(2)
    
    with col_imp1:
        st.subheader("Global Gini Feature Importance")
        imp_img_path = os.path.join(BASE_DIR, 'outputs/figures/feature_importance.png')
        if os.path.exists(imp_img_path):
            st.image(imp_img_path, use_container_width=True)
        else:
            st.info("Feature importance figure not found. Run training pipeline.")
            
    with col_imp2:
        st.subheader("SHAP TreeExplainer Summary Plot")
        shap_img_path = os.path.join(BASE_DIR, 'outputs/figures/shap_summary.png')
        if os.path.exists(shap_img_path):
            st.image(shap_img_path, use_container_width=True)
        else:
            st.info("SHAP summary plot figure not found. Run training pipeline.")
            
    st.markdown("### Top Drivers Explanation")
    st.write("""
    - **Age (Primary Driver)**: Higher age is associated with an elevated churn rate. Customers in the 45–60 bracket are substantially more likely to exit, indicating a product gap for older demographics.
    - **Number of Products (NumOfProducts)**: This exhibits a highly non-linear relationship. Customers with **2 products** have the lowest churn rates (~7%), while those with **3 products (~82%)** and **4 products (~93%)** churn at near certainty. This indicates that excessive cross-selling or service complications drive customers away.
    - **Balance**: Customers with higher balances show elevated churn risk, particularly in certain geographies. The engineered feature **BalanceSalaryRatio** highlights customers whose wealth concentration does not match their estimated salaries.
    - **Engagement Product (Engineered)**: Combining `IsActiveMember` with `NumOfProducts` shows that active members with multiple products are far more stable than inactive members with similar product counts.
    - **Geography (Germany)**: German customers churn at a rate of 32%, double the rate of French (16%) and Spanish (16%) customers. This regional anomaly indicates localized competitor strength or fee structures that prompt exits.
    """)

# ----------------- TAB 4: WHAT-IF SCENARIO SIMULATOR -----------------
with tab4:
    st.markdown("<div class='section-title'>🔮 What-if Scenario Simulator</div>", unsafe_allow_html=True)
    st.markdown("Investigate how toggling customer engagement levers (Activity Status, Product Portfolio) changes their predicted churn risk.")
    
    # Establish a default customer as base
    st.subheader("1. Base Customer Profile")
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    
    with sim_col1:
        sim_geo = st.selectbox("Geography (Base)", ["France", "Germany", "Spain"], index=1, key="sim_geo") # Germany is higher risk
        sim_gender = st.selectbox("Gender (Base)", ["Female", "Male"], index=0, key="sim_gender") # Female shows slightly higher risk
        sim_age = st.slider("Age (Base)", 18, 100, 48, key="sim_age") # Elevated age
        
    with sim_col2:
        sim_credit = st.slider("Credit Score (Base)", 350, 850, 610, key="sim_credit")
        sim_balance = st.number_input("Balance (Base, €)", 0.0, 300000.0, 110000.0, step=1000.0, key="sim_bal")
        sim_salary = st.number_input("Salary (Base, €)", 0.0, 300000.0, 95000.0, step=1000.0, key="sim_sal")
        
    with sim_col3:
        sim_tenure = st.slider("Tenure (Base)", 0, 10, 3, key="sim_tenure")
        sim_card = st.selectbox("Has Credit Card? (Base)", ["Yes", "No"], index=0, key="sim_card")
        
    sim_card_val = 1 if sim_card == "Yes" else 0
    
    st.subheader("2. Compare Engagement Levers")
    st.write("Adjust the sliders below to see how customer risk changes dynamically when their membership status or product counts are modified.")
    
    lever_col1, lever_col2 = st.columns(2)
    
    with lever_col1:
        st.markdown("#### **Base Customer State**")
        base_active = st.selectbox("Is Active Member? (Base)", ["No", "Yes"], index=0, key="b_active")
        base_products = st.slider("Number of Products (Base)", 1, 4, 3, key="b_prods")
        
    with lever_col2:
        st.markdown("#### **Scenario / Intervention State**")
        scen_active = st.selectbox("Is Active Member? (Scenario)", ["No", "Yes"], index=1, key="s_active")
        scen_products = st.slider("Number of Products (Scenario)", 1, 4, 2, key="s_prods")
        
    b_active_val = 1 if base_active == "Yes" else 0
    s_active_val = 1 if scen_active == "Yes" else 0
    
    # Run predictions
    base_customer = pd.DataFrame({
        'CreditScore': [sim_credit],
        'Geography': [sim_geo],
        'Gender': [sim_gender],
        'Age': [sim_age],
        'Tenure': [sim_tenure],
        'Balance': [sim_balance],
        'NumOfProducts': [base_products],
        'HasCrCard': [sim_card_val],
        'IsActiveMember': [b_active_val],
        'EstimatedSalary': [sim_salary]
    })
    
    scenario_customer = pd.DataFrame({
        'CreditScore': [sim_credit],
        'Geography': [sim_geo],
        'Gender': [sim_gender],
        'Age': [sim_age],
        'Tenure': [sim_tenure],
        'Balance': [sim_balance],
        'NumOfProducts': [scen_products],
        'HasCrCard': [sim_card_val],
        'IsActiveMember': [s_active_val],
        'EstimatedSalary': [sim_salary]
    })
    
    base_risk = pipeline.predict_proba(base_customer)[:, 1][0]
    scen_risk = pipeline.predict_proba(scenario_customer)[:, 1][0]
    risk_delta = scen_risk - base_risk
    
    # Output columns
    out_col1, out_col2, out_col3 = st.columns(3)
    
    with out_col1:
        st.metric(
            label="Base Risk Score",
            value=f"{base_risk:.1%}",
            delta="FLAGGED FOR CHURN" if base_risk >= threshold else "RETAINED",
            delta_color="inverse"
        )
    with out_col2:
        st.metric(
            label="Scenario Risk Score",
            value=f"{scen_risk:.1%}",
            delta="FLAGGED FOR CHURN" if scen_risk >= threshold else "RETAINED",
            delta_color="inverse"
        )
    with out_col3:
        delta_label = "Risk Reduction" if risk_delta < 0 else "Risk Increase"
        st.metric(
            label="Risk Score Delta",
            value=f"{risk_delta:+.1%}",
            delta=delta_label,
            delta_color="normal" if risk_delta > 0 else "inverse"
        )
        
    # Explain intervention impact
    if risk_delta < 0:
        st.success(f"🎉 **Positive Outcome**: Shifting the customer to the scenario state reduces their absolute churn risk by **{-risk_delta:.1%}**.")
    elif risk_delta > 0:
        st.error(f"⚠️ **Negative Outcome**: Shifting the customer to the scenario state increases their absolute churn risk by **{risk_delta:.1%}**.")
    else:
        st.info("The scenario parameters are identical to the base customer profile. Adjust sliders to see variations.")
        
    # Risk-Surface Table
    st.subheader("3. Complete Risk-Surface Heatmap (Activity × Products)")
    st.write("This table shows the predicted churn risk for all combinations of membership activity and product count, keeping other base parameters constant.")
    
    surface_rows = []
    for act_state in [0, 1]:
        row_probs = []
        for prod_count in [1, 2, 3, 4]:
            temp_customer = pd.DataFrame({
                'CreditScore': [sim_credit],
                'Geography': [sim_geo],
                'Gender': [sim_gender],
                'Age': [sim_age],
                'Tenure': [sim_tenure],
                'Balance': [sim_balance],
                'NumOfProducts': [prod_count],
                'HasCrCard': [sim_card_val],
                'IsActiveMember': [act_state],
                'EstimatedSalary': [sim_salary]
            })
            prob = pipeline.predict_proba(temp_customer)[:, 1][0]
            row_probs.append(prob)
        surface_rows.append(row_probs)
        
    surface_df = pd.DataFrame(
        surface_rows,
        index=["Inactive Member", "Active Member"],
        columns=["1 Product", "2 Products", "3 Products", "4 Products"]
    )
    
    # Styled table
    st.dataframe(
        surface_df.style.format("{:.2%}").background_gradient(
            cmap="RdYlGn_r", axis=None, vmin=0.0, vmax=1.0
        ),
        use_container_width=True
    )
    st.caption("Green represents low risk (< 20%), yellow moderate risk, red/purple represents high risk (exceeding threshold). Note how 2 Products is the low-risk sweet spot across both active and inactive members.")
