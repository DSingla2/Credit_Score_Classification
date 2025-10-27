import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import warnings

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Custom Preprocessing Functions (copied from your main script) ---
def clean_and_impute_data(df):
    df = df.copy()
    df = df.drop(columns=['Unnamed: 0', "ID", "SSN", "Name"], errors='ignore')
    def convert_to_numeric(series):
        if series.dtype == 'object':
            series = series.astype(str).str.replace("[^0-9\.]", "", regex=True)
            series = pd.to_numeric(series, errors='coerce')
        return series
    if 'Age' in df.columns:
        df['Age'] = df['Age'].astype(str).str.replace("_", "").astype("Int64", errors='ignore')
    if 'Annual_Income' in df.columns:
        df['Annual_Income'] = convert_to_numeric(df['Annual_Income'])
    if 'Num_of_Delayed_Payment' in df.columns:
        df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].str.replace("_", "").astype(float, errors='ignore')
    if 'Changed_Credit_Limit' in df.columns:
        df['Changed_Credit_Limit'] = df['Changed_Credit_Limit'].str.replace("_", "").astype(float, errors='ignore')
    if 'Outstanding_Debt' in df.columns:
        df['Outstanding_Debt'] = df['Outstanding_Debt'].str.replace("_", "").astype(float, errors='ignore')
    if 'Amount_invested_monthly' in df.columns:
        df['Amount_invested_monthly'] = df['Amount_invested_monthly'].str.replace("_", "").astype(float, errors='ignore')
    if 'Num_of_Loan' in df.columns:
        df['Num_of_Loan'] = df['Num_of_Loan'].astype(str).str.replace("_", "").astype(int, errors='ignore')
    if 'Monthly_Balance' in df.columns:
        df['Monthly_Balance'] = df['Monthly_Balance'].astype(str).str.replace("_", "").astype(float, errors='ignore')
    def convert_to_months(text):
        if pd.isna(text): return np.nan
        match = re.findall(r'\d+', str(text))
        years = int(match[0]) if len(match) > 0 else 0
        months = int(match[1]) if len(match) > 1 else 0
        return years * 12 + months
    if "Credit_History_Age" in df.columns:
        df["Credit_History_Age"] = df["Credit_History_Age"].apply(convert_to_months)
    for col in ['Age', 'Credit_History_Age', 'Num_of_Delayed_Payment',
                'Changed_Credit_Limit', 'Outstanding_Debt',
                'Num_Credit_Inquiries', 'Num_Bank_Accounts', 'Num_Credit_Card',
                'Interest_Rate', 'Total_EMI_per_month', 'Monthly_Inhand_Salary']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    if 'Age' in df.columns:
        df.loc[df['Age'] <= 0, 'Age'] = df['Age'].median()
    if 'Num_of_Loan' in df.columns:
        df.loc[df['Num_of_Loan'] < 0, 'Num_of_Loan'] = df['Num_of_Loan'].median()
    if 'Monthly_Balance' in df.columns:
        df.loc[df['Monthly_Balance'] < 0, 'Monthly_Balance'] = np.nan
    if 'Total_EMI_per_month' in df.columns:
        df.loc[df['Total_EMI_per_month'] > df['Total_EMI_per_month'].quantile(0.95), 'Total_EMI_per_month'] = np.nan
    if 'Num_of_Delayed_Payment' in df.columns:
        df.loc[df['Num_of_Delayed_Payment'] > df['Num_of_Delayed_Payment'].quantile(0.95), 'Num_of_Delayed_Payment'] = np.nan
    if 'Amount_invested_monthly' in df.columns:
        df.loc[df['Amount_invested_monthly'] > df['Amount_invested_monthly'].quantile(0.95), 'Amount_invested_monthly'] = np.nan
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in ['Credit_Mix', 'Payment_of_Min_Amount', 'Occupation', 'Payment_Behaviour']:
        if col in df.columns:
            df[col] = df[col].replace({'_': np.nan, '_______': np.nan, '!@9#%8': np.nan, 'NM': np.nan})
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    if 'Type_of_Loan' in df.columns:
        loan_types = ["Personal Loan", "Credit-Builder Loan", "Home Equity Loan",
                      "Auto Loan", "Debt Consolidation Loan", "Student Loan",
                      "Payday Loan", "Mortgage Loan"]
        df["Type_of_Loan"] = df["Type_of_Loan"].fillna("NA")
        for loan_type in loan_types:
            df[f'Has_{loan_type.replace(" ", "_")}'] = df['Type_of_Loan'].str.contains(loan_type, case=False).astype(int)
        df = df.drop('Type_of_Loan', axis=1)
    return df

def perform_final_feature_engineering(df):
    df = df.copy()
    if 'Monthly_Balance' in df.columns and 'Monthly_Inhand_Salary' in df.columns:
        df["Savings_Ratio"] = df["Monthly_Balance"] / df["Monthly_Inhand_Salary"].replace(0, 1)
    df = df.drop(columns=["Num_of_Loan", "Credit_Mix", "Monthly_Balance", "Monthly_Inhand_Salary"], errors='ignore')
    return df

def prepare_data_for_pipeline(df, X_train_cols, X_train_dtypes):
    df_cleaned = clean_and_impute_data(df.copy())
    df_engineered = perform_final_feature_engineering(df_cleaned)
    month_mapping = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
    }
    if 'Month' in df_engineered.columns:
        df_engineered["Month"] = df_engineered["Month"].map(month_mapping).astype(int, errors='ignore')
    if 'Payment_of_Min_Amount' in df_engineered.columns:
        df_engineered["Payment_of_Min_Amount"] = df_engineered["Payment_of_Min_Amount"].replace({"No": 0, "Yes": 1}).astype(int, errors='ignore')
    if 'Payment_Behaviour' in df_engineered.columns:
        df_engineered["Payment_Behaviour"] = df_engineered["Payment_Behaviour"].replace(
            {"Low_spent_Small_value_payments": 1, "Low_spent_Medium_value_payments": 2,
             "Low_spent_Large_value_payments": 3, "High_spent_Small_value_payments": 4,
             "High_spent_Medium_value_payments": 5, "High_spent_Large_value_payments": 6}
        ).astype(int, errors='ignore')
    prepared_df = pd.DataFrame(columns=X_train_cols)
    for col in prepared_df.columns:
        if col in df_engineered.columns:
            prepared_df[col] = df_engineered[col]
    for col, dtype in X_train_dtypes.items():
        if col in prepared_df.columns:
            prepared_df[col] = prepared_df[col].astype(dtype, errors='ignore')
    return prepared_df

# --- Load the Model and Metadata ---
@st.cache_resource
def load_model_and_metadata():
    """
    Loads the pre-trained pipeline and generates metadata from a quick
    preprocessing of the training data.
    """
    try:
        data = joblib.load('final_pipeline_and_metadata.joblib')
        return data['pipeline'], data['columns'], data['dtypes']
    except FileNotFoundError:
        st.error("Required file 'final_pipeline_and_metadata.joblib' not found. Please ensure it is in the same directory.")
        st.stop()

final_pipeline, X_train_cols, X_train_dtypes = load_model_and_metadata()

# --- Streamlit UI ---
st.set_page_config(page_title="Credit Score Predictor", layout="wide")
st.title("Credit Score Predictor")
st.markdown("### Predict the credit score for a new customer based on their financial profile.")

with st.form(key='credit_score_form'):
    st.subheader("Customer Information")
    col1, col2 = st.columns(2)
    
    with col1:
        customer_id = st.text_input("Customer ID", "C123456")
        month = st.selectbox("Month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
        age = st.number_input("Age", min_value=18, max_value=100, value=25)
        occupation = st.selectbox("Occupation", ['Accountant', 'Scientist', 'Journalist', 'Engineer', 'Developer', 'Lawyer', 'Manager', 'Architect', 'Musician', 'Mechanic', 'Doctor', 'Entrepreneur', 'Teacher', 'Writer', 'Media_Manager', 'Homemaker'])
        annual_income = st.number_input("Annual Income ($)", min_value=0, value=85000)
        num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, max_value=20, value=2)
        num_credit_card = st.number_input("Number of Credit Cards", min_value=0, max_value=20, value=3)
        interest_rate = st.number_input("Interest Rate (%)", min_value=0, max_value=100, value=15)
        num_of_loan = st.number_input("Number of Loans", min_value=0, max_value=20, value=4)
        type_of_loan = st.multiselect("Types of Loan (select all that apply)", ["Personal Loan", "Credit-Builder Loan", "Home Equity Loan", "Auto Loan", "Debt Consolidation Loan", "Student Loan", "Payday Loan", "Mortgage Loan"])
        payment_of_min_amount = st.selectbox("Payment of Minimum Amount", ["Yes", "No"])

    with col2:
        delay_from_due_date = st.number_input("Delay from Due Date (days)", min_value=0, max_value=100, value=12)
        num_of_delayed_payment = st.number_input("Number of Delayed Payments", min_value=0, max_value=50, value=2)
        changed_credit_limit = st.number_input("Changed Credit Limit (%)", min_value=0.0, max_value=100.0, value=10.5)
        num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0.0, max_value=50.0, value=3.0)
        credit_mix = st.selectbox("Credit Mix", ["Good", "Standard", "Poor"])
        outstanding_debt = st.number_input("Outstanding Debt ($)", min_value=0.0, value=1500.0)
        credit_utilization_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=35.6)
        credit_history_age = st.number_input("Credit History Age (in months)", min_value=0, value=150)
        total_emi_per_month = st.number_input("Total EMI per Month ($)", min_value=0.0, value=500.0)
        amount_invested_monthly = st.number_input("Amount Invested Monthly ($)", min_value=0.0, value=800.0)
        payment_behaviour = st.selectbox("Payment Behaviour", ["High_spent_Medium_value_payments", "Low_spent_Small_value_payments", "Low_spent_Medium_value_payments", "Low_spent_Large_value_payments", "High_spent_Small_value_payments", "High_spent_Large_value_payments"])
        monthly_balance = st.number_input("Monthly Balance ($)", min_value=0.0, value=4000.0)
    
    submit_button = st.form_submit_button(label='Predict Credit Score')

if submit_button:
    new_customer_data = {
        'ID': 'dummy_id',
        'Customer_ID': customer_id,
        'Month': month,
        'Name': 'dummy_name',
        'Age': str(age),
        'SSN': 'dummy_ssn',
        'Occupation': occupation,
        'Annual_Income': str(annual_income),
        'Monthly_Inhand_Salary': (annual_income / 12),
        'Num_Bank_Accounts': num_bank_accounts,
        'Num_Credit_Card': num_credit_card,
        'Interest_Rate': interest_rate,
        'Num_of_Loan': str(num_of_loan),
        'Type_of_Loan': ', '.join(type_of_loan) if type_of_loan else 'No loan',
        'Delay_from_due_date': delay_from_due_date,
        'Num_of_Delayed_Payment': str(num_of_delayed_payment),
        'Changed_Credit_Limit': str(changed_credit_limit),
        'Num_Credit_Inquiries': float(num_credit_inquiries),
        'Credit_Mix': credit_mix,
        'Outstanding_Debt': str(outstanding_debt),
        'Credit_Utilization_Ratio': float(credit_utilization_ratio),
        'Credit_History_Age': f"{int(credit_history_age / 12)} Years and {int(credit_history_age % 12)} Months",
        'Payment_of_Min_Amount': payment_of_min_amount,
        'Total_EMI_per_month': float(total_emi_per_month),
        'Amount_invested_monthly': str(amount_invested_monthly),
        'Payment_Behaviour': payment_behaviour,
        'Monthly_Balance': str(monthly_balance),
        'Credit_Score': np.nan,
        'Unnamed: 0': 0,
    }

    new_data_df = pd.DataFrame([new_customer_data])
    prepared_data = prepare_data_for_pipeline(new_data_df, X_train_cols, X_train_dtypes)

    try:
        prediction = final_pipeline.predict(prepared_data)
        reverse_score_mapping = {0: 'Good', 1: 'Standard', 2: 'Poor'}
        predicted_label = reverse_score_mapping.get(prediction[0], 'Unknown')

        st.success(f"The predicted Credit Score is: **{predicted_label}**")
        st.balloons()

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")