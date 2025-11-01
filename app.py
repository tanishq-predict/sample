import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Set page configuration
st.set_page_config(
    page_title="Coffee Shop Revenue Predictor",
    page_icon="☕",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
        color: #000000 !important;
    }
    .stApp {
        background-color: #ffffff;
        color: #000000 !important;
    }
    .css-18e3th9 {
        padding-top: 2rem;
        padding-bottom: 10rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    .css-1d391kg {
        padding-top: 3.5rem;
        padding-right: 1rem;
        padding-bottom: 3.5rem;
        padding-left: 1rem;
    }
    .reportview-container {
        background: #f0f2f6;
        color: #000000 !important;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
        color: #000000 !important;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #262730;
        color: white;
        text-align: center;
        padding: 10px 0;
    }
    h1, h2, h3, h4, h5, h6, p, div, span, label, .stMarkdown, .stText {
        color: #000000 !important;
    }
    .stDataFrame {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("☕ Coffee Shop Revenue Predictor")
st.markdown("""
This app predicts the daily revenue of a coffee shop based on various factors.
Adjust the parameters in the sidebar and click 'Predict Revenue' to see the prediction.
""")

# Sidebar with input parameters
st.sidebar.header("Input Parameters")
st.sidebar.markdown("Adjust the values below to predict revenue:")

def user_input_features():
    """Collect user input features from sidebar"""
    number_of_customers = st.sidebar.slider(
        'Number of Customers Per Day', 
        50, 500, 250,
        help="Average number of customers visiting the coffee shop per day"
    )
    
    avg_order_value = st.sidebar.slider(
        'Average Order Value ($)', 
        2.0, 10.0, 5.0, 0.1,
        help="Average amount spent by each customer per visit"
    )
    
    operating_hours = st.sidebar.slider(
        'Operating Hours Per Day', 
        6, 17, 10,
        help="Number of hours the coffee shop is open each day"
    )
    
    number_of_employees = st.sidebar.slider(
        'Number of Employees', 
        2, 14, 7,
        help="Number of employees working at the coffee shop"
    )
    
    marketing_spend = st.sidebar.slider(
        'Marketing Spend Per Day ($)', 
        10.0, 500.0, 200.0, 1.0,
        help="Daily marketing expenditure"
    )
    
    foot_traffic = st.sidebar.slider(
        'Location Foot Traffic', 
        50, 1000, 500,
        help="Estimated foot traffic in the location area"
    )
    
    data = {
        'Number_of_Customers_Per_Day': number_of_customers,
        'Average_Order_Value': avg_order_value,
        'Operating_Hours_Per_Day': operating_hours,
        'Number_of_Employees': number_of_employees,
        'Marketing_Spend_Per_Day': marketing_spend,
        'Location_Foot_Traffic': foot_traffic
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Collect user input
input_df = user_input_features()

# Main panel
st.subheader("Input Parameters")
st.write("Current parameter values:")
st.dataframe(input_df)

# Load the trained model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Load feature columns
    with open('feature_columns.pkl', 'rb') as file:
        feature_columns = pickle.load(file)
        
    # Reorder input DataFrame to match model features
    input_df = input_df[feature_columns]
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Display prediction
    st.subheader("Revenue Prediction")
    st.metric(
        label="Predicted Daily Revenue ($)", 
        value=f"${prediction[0]:,.2f}"
    )
    
    # Visualization of prediction
    st.subheader("Prediction Visualization")
    
    # Create a simple visualization
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Show the impact of customers and average order value
    revenue_components = {
        'Product Sales': input_df['Number_of_Customers_Per_Day'].iloc[0] * input_df['Average_Order_Value'].iloc[0],
        'Other Factors': prediction[0] - (input_df['Number_of_Customers_Per_Day'].iloc[0] * input_df['Average_Order_Value'].iloc[0])
    }
    
    # Plot
    ax.bar(revenue_components.keys(), revenue_components.values(), color=['#4CAF50', '#2196F3'])
    ax.set_ylabel('Revenue ($)')
    ax.set_title('Revenue Components')
    
    # Add value labels on bars
    for i, (key, value) in enumerate(revenue_components.items()):
        ax.text(i, value + 10, f'${value:.2f}', ha='center', va='bottom')
    
    st.pyplot(fig)
    
except FileNotFoundError:
    st.error("Model file not found. Please run the model_building.ipynb notebook first to train and save the model.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {str(e)}")
    st.stop()

# Feature importance visualization
try:
    # Check if model and feature_columns are defined
    if 'model' in locals() and 'feature_columns' in locals():
        # Get feature importance from the model
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=True)
        
        st.subheader("Feature Importance")
        st.write("This shows how much each feature contributes to the revenue prediction:")
        
        # Plot feature importance
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.barh(importance_df['Feature'], importance_df['Importance'], color='#FF5722')
        ax2.set_xlabel('Importance')
        ax2.set_title('Feature Importance in Revenue Prediction')
        
        # Add value labels
        for i, v in enumerate(importance_df['Importance']):
            ax2.text(v + 0.01, i, f"{v:.3f}", va='center')
        
        st.pyplot(fig2)
    else:
        st.warning("Model or feature columns not available for feature importance visualization.")
        
except Exception as e:
    st.warning("Could not display feature importance visualization.")

# Dataset information in sidebar
st.sidebar.header("Dataset Information")
st.sidebar.markdown("""
**Features used for prediction:**
- Number of Customers Per Day
- Average Order Value
- Operating Hours Per Day
- Number of Employees
- Marketing Spend Per Day
- Location Foot Traffic

**Target Variable:**
- Daily Revenue
""")

# Model information in sidebar
st.sidebar.header("Model Information")
st.sidebar.markdown("""
**Algorithm:** Random Forest Regressor

**Performance:**
- R² Score: ~0.99 (approximate)
- Very high accuracy in predicting coffee shop revenue
""")

# Additional information
st.subheader("How It Works")
st.markdown("""
The prediction is based on a Random Forest Regressor model trained on historical coffee shop data. 
The model takes into account various factors that influence revenue and provides an estimate based on those inputs.

**Key factors affecting revenue:**
1. **Number of Customers:** More customers generally mean higher revenue
2. **Average Order Value:** Higher spending per customer increases revenue
3. **Location Foot Traffic:** Better locations attract more customers
4. **Marketing Spend:** Investment in marketing can increase customer visits
5. **Operating Hours:** Longer hours may capture more customers
6. **Number of Employees:** Adequate staffing improves service quality
""")

# Footer
st.markdown("""
<div class="footer">
    Coffee Shop Revenue Predictor App | Machine Learning Project
</div>
""", unsafe_allow_html=True)
