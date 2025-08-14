import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import io
import matplotlib.pyplot as plt # Keep for general plot cleanup if needed

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for enhanced UI and Dark Theme ---
st.markdown(f"""
<style>
    :root {{ /* Dark Theme variables (default and only theme) */
        --primary-bg: #1a1a2e; /* Dark blue/black */
        --secondary-bg: #16213E; /* Slightly lighter dark blue for cards */
        --text-color: #f0f2f6; /* Light gray for main text */
        --header-color: #e0e0e0; /* Lighter header text */
        --metric-card-bg: rgba(255, 255, 255, 0.1); /* Slightly transparent white for metric cards */
        --metric-card-text: #f0f2f6; /* Text inside metric cards */
        --metric-card-h3: #FF7B8C; /* Accent color for metric card titles */
        --button-bg: #E94560; /* Button background */
        --button-hover-bg: #FF7B8C; /* Button hover background */
        --button-text: white; /* Button text color */
        --sidebar-bg: #0F3460; /* Sidebar background */
        --sidebar-text: white; /* Sidebar text color */
        --tab-text: #e0e0e0; /* Tab text color */
        --prediction-bg: #2d3e50; /* Darker blue for prediction box */
        --prediction-border: #3c526a; /* Border for prediction box */
        --prediction-text: #f0f2f6; /* Text in prediction box */
        --input-border: #3c526a; /* Darker border for inputs */
        --input-text: #f0f2f6; /* Light text for inputs */
        --input-bg: #16213E; /* Input background */
        --churn-color: #E94560; /* Red for churned */
        --no-churn-color: #00B050; /* Green for non-churned */
        --accent-color-gold: #FFD700; /* Gold accent color for other charts */
    }}

    /* Apply theme variables to Streamlit's main content area */
    .streamlit-container {{ /* This targets the main content area effectively */
        background-color: var(--primary-bg) !important; /* Use !important to override Streamlit defaults */
        color: var(--text-color);
    }}
    .stApp {{ /* Also apply to the main Streamlit app div */
        background-color: var(--primary-bg) !important;
        color: var(--text-color);
    }}
    .main-header {{
        font-size: 4.8em; /* Even Larger font size for main header */
        color: var(--header-color);
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.2em;
        text-shadow: 4px 4px 8px rgba(0,0,0,0.4); /* More pronounced shadow */
        background: linear-gradient(45deg, #E94560, #FF7B8C, #FFD700); /* More vibrant gradient */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }}
    .sub-header {{
        font-size: 2.4em; /* Slightly larger sub-header */
        color: var(--text-color);
        text-align: center;
        margin-bottom: 1.8em;
        font-style: italic;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: var(--header-color);
    }}
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
        color: var(--tab-text);
    }}
    .data-summary, .section-card {{
        background-color: var(--secondary-bg);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }}
    .description-text {{
        color: var(--text-color);
    }}
    .metrics-container {{
        display: flex;
        justify-content: space-around;
        gap: 20px;
        margin-bottom: 30px;
        flex-wrap: wrap;
    }}
    .metric-card {{
        background-color: var(--metric-card-bg);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        text-align: center;
        flex: 1;
        min-width: 200px;
        margin: 10px;
    }}
    .metric-card h3 {{
        color: var(--metric-card-h3);
        font-size: 1.5em;
        margin-bottom: 10px;
    }}
    .metric-card p {{
        font-size: 2em;
        font-weight: bold;
        color: var(--metric-card-text);
    }}
    .stTextInput label, .stSelectbox label, .stSlider label {{
        font-weight: bold;
        color: var(--header-color);
    }}
    .stTextInput input, .stSelectbox div[role="listbox"], .stSlider div[data-testid="stRealSlider"] {{
        border-radius: 5px;
        border: 1px solid var(--input-border);
        padding: 8px;
        color: var(--input-text);
        background-color: var(--input-bg);
    }}
    .prediction-result {{
        background-color: var(--prediction-bg);
        border: 1px solid var(--prediction-border);
        padding: 15px;
        border-radius: 8px;
        font-size: 1.2em;
        font-weight: bold;
        color: var(--prediction-text);
        text-align: center;
        margin-top: 20px;
    }}
    .stFileUploader > div > button, .stButton > button {{
        background-color: var(--button-bg);
        color: var(--button-text);
    }}
    .stFileUploader > div > button:hover, .stButton > button:hover {{
        background-color: var(--button-hover-bg);
    }}

    /* Style for sidebar itself */
    .st-emotion-cache-vk337f {{ /* This targets the sidebar container specifically */
        background-color: var(--sidebar-bg);
    }}
    .st-emotion-cache-1tmx63h, .st-emotion-cache-1tmx63h p, .st-emotion-cache-1tmx63h label {{ /* Sidebar content, paragraphs, and labels */
        color: var(--sidebar-text);
    }}
    .st-emotion-cache-1tmx63h h1,
    .st-emotion-cache-1tmx63h h2,
    .st-emotion-cache-1tmx63h h3,
    .st-emotion-cache-1tmx63h h4,
    .st-emotion-cache-1tmx63h h5,
    .st-emotion-cache-1tmx63h h6 {{
        color: var(--sidebar-text); /* Ensure sidebar headers are visible */
    }}

    /* Adjust Streamlit's internal radio button labels (if any remain) */
    .stRadio > label {{
        color: var(--text-color); /* Main radio button labels */
    }}
    .stRadio div[role="radiogroup"] label {{
        color: var(--text-color); /* Individual radio option labels */
    }}

</style>
""", unsafe_allow_html=True)

# --- Removed JS injection for theme class as it's no longer needed for dynamic switching ---


# --- Helper Functions ---
@st.cache_data
def load_data(uploaded_file):
    """Loads data from CSV or Excel file."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

@st.cache_data
def preprocess_data(df, is_predicting_single_instance=False, trained_columns=None):
    """
    Preprocesses the dataframe: handles 'Total Charges' conversion,
    encodes categorical features, and fills missing 'Churn Reason'.
    If is_predicting_single_instance is True, it ensures alignment with trained_columns.
    """
    df = df.copy()

    # Handle 'Total Charges' - convert to numeric, coercing errors to NaN
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    df['Total Charges'].fillna(0, inplace=True)

    # Fill missing 'Churn Reason' with 'No Churn' for non-churned customers,
    # and 'Unknown' for churned customers with no specified reason.
    if 'Churn Label' in df.columns:
        df['Churn Reason'] = df.apply(
            lambda row: 'No Churn' if row['Churn Label'] == 'No' else (
                row['Churn Reason'] if pd.notnull(row['Churn Reason']) else 'Unknown Churn Reason'
            ), axis=1
        )
    else:
        # Fallback if 'Churn Value' is present
        if 'Churn Value' in df.columns:
            df['Churn Reason'] = df.apply(
                lambda row: 'No Churn' if row['Churn Value'] == 0 else (
                    row['Churn Reason'] if pd.notnull(row['Churn Reason']) else 'Unknown Churn Reason'
                ), axis=1
            )

    # If predicting a single instance, we need a special handling for one-hot encoding
    # to ensure all expected dummy columns are present, even if the single instance
    # doesn't contain all categories.
    if is_predicting_single_instance and trained_columns is not None:
        # Create a template DataFrame with all trained columns, initialized to 0
        processed_df_aligned = pd.DataFrame(0, index=[0], columns=trained_columns)

        # Populate numerical features directly
        for col in df.select_dtypes(include=np.number).columns:
            if col in processed_df_aligned.columns:
                processed_df_aligned.loc[0, col] = df.loc[0, col]

        # Handle categorical features by setting the appropriate one-hot encoded column to 1
        categorical_cols_for_encoding = [
            'Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service',
            'Multiple Lines', 'Internet Service', 'Online Security', 'Online Backup',
            'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies',
            'Contract', 'Paperless Billing', 'Payment Method'
        ]
        
        for cat_col in categorical_cols_for_encoding:
            if cat_col in df.columns:
                value = df.loc[0, cat_col]
                
                # Handle binary mappings first
                if cat_col == 'Gender':
                    # Assuming 'Male' maps to 1, 'Female' to 0 if Gender_Female was dropped
                    # Or if Gender_Male was dropped and Gender_Female is 1
                    # This needs to be consistent with how get_dummies created columns
                    if 'Gender_Male' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, 'Gender_Male'] = 1 if value == 'Male' else 0
                    elif 'Gender_Female' in processed_df_aligned.columns: # Fallback if 'Male' was dropped
                        processed_df_aligned.loc[0, 'Gender_Female'] = 1 if value == 'Female' else 0

                elif cat_col == 'Senior Citizen':
                    if 'Senior Citizen_Yes' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, 'Senior Citizen_Yes'] = 1 if value == 'Yes' else 0
                elif cat_col == 'Partner':
                    if 'Partner_Yes' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, 'Partner_Yes'] = 1 if value == 'Yes' else 0
                elif cat_col == 'Dependents':
                    if 'Dependents_Yes' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, 'Dependents_Yes'] = 1 if value == 'Yes' else 0
                elif cat_col == 'Phone Service':
                    if 'Phone Service_Yes' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, 'Phone Service_Yes'] = 1 if value == 'Yes' else 0
                elif cat_col == 'Paperless Billing':
                    if 'Paperless Billing_Yes' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, 'Paperless Billing_Yes'] = 1 if value == 'Yes' else 0
                elif cat_col == 'Multiple Lines':
                    if value == 'Yes' and 'Multiple Lines_Yes' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, 'Multiple Lines_Yes'] = 1
                    elif value == 'No phone service' and 'Multiple Lines_No phone service' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, 'Multiple Lines_No phone service'] = 1 # This might not be dropped if it's the first category
                elif cat_col == 'Internet Service':
                    if f'Internet Service_{value}' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, f'Internet Service_{value}'] = 1
                elif cat_col == 'Online Security':
                    if value == 'Yes' and 'Online Security_Yes' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, 'Online Security_Yes'] = 1
                    elif value == 'No internet service' and 'Online Security_No internet service' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, 'Online Security_No internet service'] = 1
                elif cat_col == 'Online Backup':
                    if value == 'Yes' and 'Online Backup_Yes' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, 'Online Backup_Yes'] = 1
                    elif value == 'No internet service' and 'Online Backup_No internet service' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, 'Online Backup_No internet service'] = 1
                elif cat_col == 'Device Protection':
                    if value == 'Yes' and 'Device Protection_Yes' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, 'Device Protection_Yes'] = 1
                    elif value == 'No internet service' and 'Device Protection_No internet service' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, 'Device Protection_No internet service'] = 1
                elif cat_col == 'Tech Support':
                    if value == 'Yes' and 'Tech Support_Yes' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, 'Tech Support_Yes'] = 1
                    elif value == 'No internet service' and 'Tech Support_No internet service' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, 'Tech Support_No internet service'] = 1
                elif cat_col == 'Streaming TV':
                    if value == 'Yes' and 'Streaming TV_Yes' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, 'Streaming TV_Yes'] = 1
                    elif value == 'No internet service' and 'Streaming TV_No internet service' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, 'Streaming TV_No internet service'] = 1
                elif cat_col == 'Streaming Movies':
                    if value == 'Yes' and 'Streaming Movies_Yes' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, 'Streaming Movies_Yes'] = 1
                    elif value == 'No internet service' and 'Streaming Movies_No internet service' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, 'Streaming Movies_No internet service'] = 1
                elif cat_col == 'Contract':
                    if value == 'One year' and 'Contract_One year' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, 'Contract_One year'] = 1
                    elif value == 'Two year' and 'Contract_Two year' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, 'Contract_Two year'] = 1
                elif cat_col == 'Payment Method':
                    if f'Payment Method_{value}' in processed_df_aligned.columns:
                        processed_df_aligned.loc[0, f'Payment Method_{value}'] = 1
        
        df = processed_df_aligned # The aligned dataframe is now the processed one
        df.fillna(df.median(), inplace=True) # Final fill for any remaining NaNs

    else: # Normal preprocessing for full dataset (training/EDA)
        for column in df.columns:
            if df[column].dtype == 'object':
                if df[column].nunique() == 2:
                    unique_vals = df[column].unique()
                    if 'Yes' in unique_vals and 'No' in unique_vals:
                        df[column] = df[column].map({'Yes': 1, 'No': 0})
                    elif 'Male' in unique_vals and 'Female' in unique_vals:
                        df[column] = df[column].map({'Male': 1, 'Female': 0})
                    elif 'No' in unique_vals and 'No internet service' in unique_vals:
                        df[column] = df[column].map({'No internet service': 0, 'No': 0, 'Yes': 1})
                    elif 'No' in unique_vals and 'No phone service' in unique_vals:
                        df[column] = df[column].map({'No phone service': 0, 'No': 0, 'Yes': 1})
                    elif 'Month-to-month' in unique_vals:
                        df[column] = df[column].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
                    else:
                        le = LabelEncoder()
                        df[column] = le.fit_transform(df[column])
                elif column not in ['CustomerID', 'Lat Long', 'Churn Label', 'Churn Reason']:
                    df = pd.get_dummies(df, columns=[column], prefix=column, drop_first=True)

        # Ensure all feature columns are numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    df = df.drop(columns=[col]) # Drop if cannot convert to numeric

        df = df.select_dtypes(include=np.number) # Select only numeric columns

        # Drop columns with all NaNs if any resulted from conversion
        df.dropna(axis=1, how='all', inplace=True)
        # Fill remaining NaNs (if any) with median for numerical stability
        df.fillna(df.median(), inplace=True)

    return df

@st.cache_data
def train_model_and_get_feature_importance(df):
    """
    Trains a RandomForestClassifier and calculates feature importances.
    Assumes 'Churn Value' is the target variable (0 for No Churn, 1 for Churn).
    Returns the trained model, feature importances, and the test set for accuracy calculation.
    """
    if 'Churn Value' not in df.columns:
        st.error("The dataset must contain a 'Churn Value' column (0 for No Churn, 1 for Churn) for feature importance analysis.")
        return None, None, None, None

    features = [col for col in df.columns if col not in ['CustomerID', 'Churn Label', 'Churn Value', 'Churn Score', 'CLTV', 'Churn Reason', 'Count', 'Zip Code', 'Lat Long']]
    X = df[features]
    y = df['Churn Value']

    # Ensure all feature columns are numeric (this step is also in preprocess_data, but good to double check)
    X = X.select_dtypes(include=np.number)
    X.dropna(axis=1, how='all', inplace=True)
    X.fillna(X.median(), inplace=True)

    if X.empty:
        st.error("No numeric features remaining after preprocessing. Cannot train model.")
        return None, None, None, None

    if y.nunique() < 2:
        st.warning("Target variable 'Churn Value' has less than 2 unique classes after splitting. Cannot train model.")
        return None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    feature_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    return model, feature_importances, X_test, y_test


# --- Main App Logic ---
def app():
    st.markdown("<h1 class='main-header'>Customer Churn Prediction & Analysis </h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Uncover insights from your customer data and predict churn.</p>", unsafe_allow_html=True)

    st.sidebar.header("Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

    # Initialize variables for broader scope before conditional blocks
    df = None
    original_df_copy = None
    df_processed = None
    model = None
    feature_importances = None
    X_test = None
    y_test = None

    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.sidebar.success("File uploaded successfully! 🎉")
            st.sidebar.write(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

            original_df_copy = df.copy() # Defined here after successful load

            # Preprocess the data for analysis and model training
            df_processed = preprocess_data(df.copy()) # Pass a copy to avoid modifying original df

            # Train model and get relevant outputs once for the entire app run
            model, feature_importances, X_test, y_test = train_model_and_get_feature_importance(df_processed)

            if model is None: # If training failed for any reason
                st.error("Model could not be trained. Please check the dataset requirements (e.g., 'Churn Value' column, sufficient numeric features).")

        else:
            st.sidebar.error("Failed to load file.")
    else:
        st.info("Please upload a dataset to get started. The dataset should ideally contain 'Churn Value' (0 or 1) and other customer demographic/service details.")

    # Only show tabs if data is loaded and model is trained successfully
    if df is not None and model is not None:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview & EDA", "💡 Feature Importance", "🔍 Deep Dive Analysis", "🚀 Predictions", "🧪 What-If Analysis"])

        with tab1:
            st.header("Dataset Overview & Exploratory Data Analysis")

            st.markdown("---")

            # Display key metrics in horizontal cards
            if 'Churn Value' in df.columns:
                total_customers = len(df)
                churned_customers = df['Churn Value'].sum()
                active_customers = total_customers - churned_customers
                churn_rate = (churned_customers / total_customers) * 100 if total_customers > 0 else 0

                st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><h3>Total Customers</h3><p>{total_customers}</p></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><h3>Active Customers</h3><p>{active_customers}</p></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><h3>Churned Customers</h3><p>{churned_customers}</p></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><h3>Churn Rate</h3><p>{churn_rate:.2f}%</p></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("Dataset Description")
            st.markdown("<div class='section-card description-text'>", unsafe_allow_html=True)
            st.write(f"The dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
            st.write("It includes various demographic, service, and billing information for customers, along with their churn status.")
            st.write("The primary goal of analyzing this data is to identify factors contributing to customer churn and ultimately to develop strategies for retention.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("Data Head")
            st.dataframe(df.head())

            st.subheader("Missing Values")
            missing_df = df.isnull().sum().reset_index()
            missing_df.columns = ['Column', 'Missing Count']
            missing_df['Missing Percentage'] = (missing_df['Missing Count'] / len(df)) * 100
            st.dataframe(missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False))
            if missing_df['Missing Count'].sum() == 0:
                st.success("No missing values found in the dataset! 🎉")
            else:
                st.warning("Missing values detected. These will be handled during preprocessing for analysis.")

            st.subheader("Data Types")
            st.dataframe(df.dtypes.reset_index().rename(columns={0: 'Data Type', 'index': 'Column'}))

            st.subheader("Distribution of Churn")
            if 'Churn Value' in df.columns:
                churn_counts = df['Churn Value'].value_counts().reset_index()
                churn_counts.columns = ['Churn Value', 'Count']
                fig_churn = px.pie(churn_counts, names='Churn Value', values='Count',
                                   title='Distribution of Customer Churn (0: No Churn, 1: Churn)',
                                   color_discrete_map={0: '#00B050', 1: '#E94560'}) # Explicit colors
                st.plotly_chart(fig_churn, use_container_width=True)
            else:
                st.warning("Cannot plot churn distribution: 'Churn Value' column not found.")

            st.subheader("Categorical Feature Distributions vs. Churn")
            categorical_cols = [
                'Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service',
                'Multiple Lines', 'Internet Service', 'Online Security', 'Online Backup',
                'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies',
                'Contract', 'Paperless Billing', 'Payment Method'
            ]
            categorical_cols = [col for col in categorical_cols if col in df.columns]

            if categorical_cols:
                selected_categorical_col = st.selectbox("Select a categorical feature to visualize:", categorical_cols)

                if selected_categorical_col and 'Churn Value' in df.columns:
                    churn_by_cat = df.groupby([selected_categorical_col, 'Churn Value']).size().unstack(fill_value=0)
                    churn_by_cat = churn_by_cat.reset_index()
                    churn_by_cat.columns = [selected_categorical_col, 'No Churn', 'Churn']
                    churn_by_cat_melted = churn_by_cat.melt(id_vars=selected_categorical_col, var_name='Churn Status', value_name='Count')

                    fig_cat_churn = px.bar(churn_by_cat_melted, x=selected_categorical_col, y='Count', color='Churn Status',
                                           title=f'Churn Distribution by {selected_categorical_col}',
                                           barmode='group',
                                           color_discrete_map={'No Churn': '#00B050', 'Churn': '#E94560'}) # Explicit colors
                    st.plotly_chart(fig_cat_churn, use_container_width=True)
                else:
                    st.warning("Cannot visualize categorical features: 'Churn Value' column not found or no categorical feature selected.")
            else:
                st.info("No common categorical features found for detailed distribution analysis.")

            st.subheader("Numerical Feature Distributions")
            numerical_cols = [
                'Tenure Months', 'Monthly Charges', 'Total Charges', 'CLTV', 'Churn Score'
            ]
            numerical_cols = [col for col in numerical_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

            if numerical_cols:
                selected_numerical_col = st.selectbox("Select a numerical feature to visualize:", numerical_cols)

                if selected_numerical_col:
                    fig_hist = px.histogram(df, x=selected_numerical_col, marginal="box",
                                            title=f'Distribution of {selected_numerical_col}',
                                            color_discrete_sequence=['#00B050']) # Explicit color
                    st.plotly_chart(fig_hist, use_container_width=True)

                    if 'Churn Value' in df.columns:
                        fig_box = px.box(df, x='Churn Value', y=selected_numerical_col,
                                         title=f'{selected_numerical_col} by Churn Status',
                                         color='Churn Value',
                                         color_discrete_map={0: '#00B050', 1: '#E94560'}) # Explicit colors
                        st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("No common numerical features found for detailed distribution analysis.")


        with tab2:
            st.header("Feature Importance & Model Insights")
            st.markdown("Understanding which factors drive churn is crucial for effective retention strategies.")

            if model and feature_importances is not None:
                st.subheader("Top 10 Feature Importance (from RandomForest Model)")
                fig_fi = px.bar(feature_importances.head(10), x='importance', y='feature', orientation='h',
                                title='Top 10 Features Influencing Churn',
                                color_discrete_sequence=['#FFD700']) # Changed to Gold
                fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_fi, use_container_width=True)

                st.markdown("""
                <div class='section-card description-text'>
                <p>The **Feature Importance** chart above displays the relative importance of each feature in predicting customer churn, as determined by a Random Forest Classifier. A higher bar indicates that the feature plays a more significant role in the model's decision-making process. This helps us understand which aspects of customer behavior or demographics are most strongly associated with churn.</p>
                <p>Commonly, features like **Tenure Months**, **Contract Type**, **Monthly Charges**, and **Internet Service** (especially Fiber Optic) are high indicators of churn risk.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Feature importance analysis requires a trained model. Please ensure 'Churn Value' column is present and valid numeric features exist in your dataset.")


        with tab3:
            st.header("Deep Dive Analysis & Business Insights")
            st.markdown("Here we can explore specific aspects of churn based on the dataset and identified key factors.")

            if 'Churn Reason' in df.columns and 'Churn Label' in df.columns:
                st.subheader("Top Churn Reasons")
                churn_reasons_df = df[df['Churn Label'] == 'Yes']
                if not churn_reasons_df.empty:
                    churn_reasons = churn_reasons_df['Churn Reason'].value_counts().head(10).reset_index()
                    churn_reasons.columns = ['Reason', 'Count']
                    fig_reasons = px.bar(churn_reasons, x='Count', y='Reason', orientation='h',
                                         title='Top 10 Reasons for Churn (among churned customers)',
                                         color_discrete_sequence=['#FFD700']) # Changed to Gold
                    fig_reasons.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_reasons, use_container_width=True)
                else:
                    st.info("No churned customers found to display churn reasons.")


                st.markdown("""
                <div class='section-card description-text'>
                <p>Understanding the **Churn Reasons** directly from the data (if available) provides actionable insights. This chart highlights the most frequently cited reasons for customers deciding to leave. Businesses can leverage this information to address specific pain points and improve service quality or offerings.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("The 'Churn Reason' or 'Churn Label' column was not found in your dataset. Cannot display churn reasons.")


            if 'Tenure Months' in df.columns and 'Churn Value' in df.columns:
                st.subheader("Churn Rate by Tenure Group")
                bins = [0, 12, 24, 36, 48, 60, df['Tenure Months'].max() + 1]
                labels = ['0-12 Months', '13-24 Months', '25-36 Months', '37-48 Months', '49-60 Months', '60+ Months']
                df['Tenure Group'] = pd.cut(df['Tenure Months'], bins=bins, labels=labels, right=False)

                churn_by_tenure = df.groupby('Tenure Group')['Churn Value'].value_counts(normalize=True).unstack().fillna(0)
                if 1 in churn_by_tenure.columns:
                    churn_by_tenure['Churn Rate (%)'] = churn_by_tenure[1] * 100
                    fig_tenure_churn = px.bar(churn_by_tenure.reset_index(), x='Tenure Group', y='Churn Rate (%)',
                                              title='Churn Rate by Tenure Group',
                                              color_discrete_sequence=['#E94560']) # Explicit color
                    st.plotly_chart(fig_tenure_churn, use_container_width=True)
                else:
                    st.info("No churned customers found to calculate churn rate by tenure.")
            else:
                st.warning("Cannot analyze churn by tenure: 'Tenure Months' or 'Churn Value' column not found.")

            if 'Contract' in df.columns and 'Churn Value' in df.columns:
                st.subheader("Churn Rate by Contract Type")
                churn_by_contract = df.groupby('Contract')['Churn Value'].value_counts(normalize=True).unstack().fillna(0)
                if 1 in churn_by_contract.columns:
                    churn_by_contract['Churn Rate (%)'] = churn_by_contract[1] * 100
                    fig_contract_churn = px.bar(churn_by_contract.reset_index(), x='Contract', y='Churn Rate (%)',
                                                title='Churn Rate by Contract Type',
                                                color_discrete_sequence=['#E94560']) # Explicit color
                    st.plotly_chart(fig_contract_churn, use_container_width=True)
                else:
                    st.info("No churned customers found to calculate churn rate by contract type.")
            else:
                st.warning("Cannot analyze churn by contract type: 'Contract' or 'Churn Value' column not found.")

            st.subheader("Further Exploration")
            st.write("Here are some ideas for further analysis you might want to explore:")
            st.markdown("- **Geographic Churn:** If 'Latitude' and 'Longitude' are reliable, you could plot churned customers on a map.")
            st.markdown("- **CLTV vs Churn:** Analyze how Customer Lifetime Value relates to churn.")
            st.markdown("- **Monthly Charges vs Churn:** Scatter plot of monthly charges vs. churn.")

        with tab4:
            st.header("Model Predictions & Performance")
            st.markdown("Here you can view the model's predictions and download the results.")

            if model and X_test is not None and y_test is not None:
                st.subheader("Model Accuracy")
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.success(f"The **Accuracy Score** of the RandomForest model on the test set is: **{accuracy*100:.0f}%**") # Changed to whole percentage
                st.info("Accuracy indicates the proportion of correctly classified instances. For imbalanced datasets, other metrics like Precision, Recall, or F1-score might be more informative.")

                st.subheader("Download Predictions")
                st.write("Click the button below to download a CSV file containing the original data with added churn predictions.")

                # Ensure X_predict is aligned with the model's training features
                # 1. Preprocess the original_df_copy to get the same transformations (including one-hot encoding)
                temp_processed_df_for_prediction = preprocess_data(original_df_copy.copy())

                # 2. Select only the numeric columns from the processed df
                temp_processed_df_for_prediction = temp_processed_df_for_prediction.select_dtypes(include=np.number)

                # 3. Align columns with the model's training features (model.feature_names_in_)
                # Use reindex to create a DataFrame with all columns the model expects, filling missing with 0
                X_predict_aligned = temp_processed_df_for_prediction.reindex(columns=model.feature_names_in_, fill_value=0)

                # 4. Fill any remaining NaNs (should be minimal after previous steps if data is consistent)
                X_predict_aligned.fillna(X_predict_aligned.median(), inplace=True) # Using median for robustness

                if not X_predict_aligned.empty and X_predict_aligned.shape[1] == len(model.feature_names_in_):
                    try:
                        churn_probabilities = model.predict_proba(X_predict_aligned)[:, 1] # Probability of churn (class 1)
                        churn_predictions = model.predict(X_predict_aligned) # Binary prediction (0 or 1)

                        # Add predictions to a copy of the original (unprocessed) DataFrame
                        results_df = original_df_copy.copy()
                        results_df['Predicted_Churn_Value'] = churn_predictions
                        results_df['Predicted_Churn_Probability'] = churn_probabilities
                        results_df['Predicted_Churn_Label'] = results_df['Predicted_Churn_Value'].map({1: 'Yes', 0: 'No'})

                        csv_buffer = io.StringIO()
                        results_df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue().encode('utf-8')

                        st.download_button(
                            label="Download predictions.csv",
                            data=csv_data,
                            file_name="churn_predictions.csv",
                            mime="text/csv",
                            help="Download the dataset with predicted churn values and probabilities."
                        )
                        st.dataframe(results_df.head()) # Show a preview of the results
                    except Exception as e:
                        st.error(f"An error occurred during prediction generation: {e}")
                        st.warning("Please ensure your dataset is consistent with the data used for model training (e.g., all expected features are present after preprocessing).")
                else:
                    st.warning("Cannot generate predictions: Aligned feature set is empty or column count mismatch after preprocessing and alignment. This can happen if required features are missing or entirely different after preprocessing.")
            else:
                st.warning("Model is not trained. Please ensure a dataset is uploaded and contains the 'Churn Value' column to enable predictions.")

        with tab5: # New "What-If" Analysis Tab
            st.header("🧪 What-If Scenario Analysis")
            st.markdown("""
            <div class='section-card description-text'>
            <p>Ever wondered how changes in customer attributes could affect their churn probability? Use this interactive tool to simulate different customer profiles and see your model's prediction!</p>
            <p>Adjust the sliders and select options below to create a hypothetical customer, then click 'Predict Churn' to see the outcome.</p>
            </div>
            """, unsafe_allow_html=True)

            if model and model.feature_names_in_ is not None and df_processed is not None: # Use df_processed here
                st.subheader("Define a Hypothetical Customer Profile")

                # Collect feature inputs
                input_data = {}
                
                # Get unique values for selectboxes dynamically from the original data
                # Using df.get() with default empty list to avoid KeyError if column is missing
                # Ensure these options are derived from the *original* df to show human-readable values
                gender_options = df['Gender'].unique().tolist() if 'Gender' in df.columns else ['Female', 'Male']
                senior_citizen_options = df['Senior Citizen'].unique().tolist() if 'Senior Citizen' in df.columns else ['No', 'Yes']
                partner_options = df['Partner'].unique().tolist() if 'Partner' in df.columns else ['Yes', 'No']
                dependents_options = df['Dependents'].unique().tolist() if 'Dependents' in df.columns else ['Yes', 'No']
                phone_service_options = df['Phone Service'].unique().tolist() if 'Phone Service' in df.columns else ['Yes', 'No']
                multiple_lines_options = df['Multiple Lines'].unique().tolist() if 'Multiple Lines' in df.columns else ['No phone service', 'No', 'Yes']
                internet_service_options = df['Internet Service'].unique().tolist() if 'Internet Service' in df.columns else ['Fiber optic', 'DSL', 'No']
                online_security_options = df['Online Security'].unique().tolist() if 'Online Security' in df.columns else ['Yes', 'No', 'No internet service']
                online_backup_options = df['Online Backup'].unique().tolist() if 'Online Backup' in df.columns else ['Yes', 'No', 'No internet service']
                device_protection_options = df['Device Protection'].unique().tolist() if 'Device Protection' in df.columns else ['Yes', 'No', 'No internet service']
                tech_support_options = df['Tech Support'].unique().tolist() if 'Tech Support' in df.columns else ['Yes', 'No', 'No internet service']
                streaming_tv_options = df['Streaming TV'].unique().tolist() if 'Streaming TV' in df.columns else ['Yes', 'No', 'No internet service']
                streaming_movies_options = df['Streaming Movies'].unique().tolist() if 'Streaming Movies' in df.columns else ['Yes', 'No', 'No internet service']
                contract_options = df['Contract'].unique().tolist() if 'Contract' in df.columns else ['Month-to-month', 'One year', 'Two year']
                paperless_billing_options = df['Paperless Billing'].unique().tolist() if 'Paperless Billing' in df.columns else ['Yes', 'No']
                payment_method_options = df['Payment Method'].unique().tolist() if 'Payment Method' in df.columns else ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']

                # Set default values for sliders based on dataset min/max if available
                # CRITICAL FIX: Use df_processed for numerical stats to ensure they are float/int
                min_tenure = df_processed['Tenure Months'].min() if 'Tenure Months' in df_processed.columns else 0
                max_tenure = df_processed['Tenure Months'].max() if 'Tenure Months' in df_processed.columns else 72
                avg_tenure = int(df_processed['Tenure Months'].mean()) if 'Tenure Months' in df_processed.columns else 29

                min_monthly = df_processed['Monthly Charges'].min() if 'Monthly Charges' in df_processed.columns else 18.0
                max_monthly = df_processed['Monthly Charges'].max() if 'Monthly Charges' in df_processed.columns else 120.0
                avg_monthly = float(df_processed['Monthly Charges'].mean()) if 'Monthly Charges' in df_processed.columns else 70.0

                min_total = df_processed['Total Charges'].min() if 'Total Charges' in df_processed.columns else 0.0
                max_total = df_processed['Total Charges'].max() if 'Total Charges' in df_processed.columns else 8000.0
                avg_total = float(df_processed['Total Charges'].mean()) if 'Total Charges' in df_processed.columns else 2000.0


                input_data['Tenure Months'] = st.slider('Tenure in Months', min_value=min_tenure, max_value=max_tenure, value=avg_tenure)
                input_data['Monthly Charges'] = st.slider('Monthly Charges', min_value=min_monthly, max_value=max_monthly, value=avg_monthly)
                input_data['Total Charges'] = st.slider('Total Charges', min_value=min_total, max_value=max_total, value=avg_total)
                
                # Check for existence of columns before creating selectboxes
                if 'Gender' in df.columns: input_data['Gender'] = st.selectbox('Gender', gender_options)
                if 'Senior Citizen' in df.columns: input_data['Senior Citizen'] = st.selectbox('Senior Citizen', senior_citizen_options)
                if 'Partner' in df.columns: input_data['Partner'] = st.selectbox('Partner', partner_options)
                if 'Dependents' in df.columns: input_data['Dependents'] = st.selectbox('Dependents', dependents_options)
                if 'Phone Service' in df.columns: input_data['Phone Service'] = st.selectbox('Phone Service', phone_service_options)
                if 'Multiple Lines' in df.columns: input_data['Multiple Lines'] = st.selectbox('Multiple Lines', multiple_lines_options)
                if 'Internet Service' in df.columns: input_data['Internet Service'] = st.selectbox('Internet Service', internet_service_options)
                if 'Online Security' in df.columns: input_data['Online Security'] = st.selectbox('Online Security', online_security_options)
                if 'Online Backup' in df.columns: input_data['Online Backup'] = st.selectbox('Online Backup', online_backup_options)
                if 'Device Protection' in df.columns: input_data['Device Protection'] = st.selectbox('Device Protection', device_protection_options)
                if 'Tech Support' in df.columns: input_data['Tech Support'] = st.selectbox('Tech Support', tech_support_options) # Corrected to check df.columns
                if 'Streaming TV' in df.columns: input_data['Streaming TV'] = st.selectbox('Streaming TV', streaming_tv_options)
                if 'Streaming Movies' in df.columns: input_data['Streaming Movies'] = st.selectbox('Streaming Movies', streaming_movies_options)
                if 'Contract' in df.columns: input_data['Contract'] = st.selectbox('Contract', contract_options)
                if 'Paperless Billing' in df.columns: input_data['Paperless Billing'] = st.selectbox('Paperless Billing', paperless_billing_options)
                if 'Payment Method' in df.columns: input_data['Payment Method'] = st.selectbox('Payment Method', payment_method_options)
                # Add other relevant features here as per your data if you want to allow user to control them

                # Create a DataFrame from the input for prediction
                if input_data:
                    # Construct a raw DataFrame that resembles original row structure
                    single_instance_df = pd.DataFrame([input_data])
                    
                    # Preprocess this single instance using the specialized mode
                    # This ensures it aligns perfectly with the model's trained features
                    processed_single_instance = preprocess_data(
                        single_instance_df,
                        is_predicting_single_instance=True,
                        trained_columns=model.feature_names_in_
                    )

                    if st.button("Predict Churn for This Profile"):
                        if not processed_single_instance.empty and processed_single_instance.shape[1] == len(model.feature_names_in_):
                            try:
                                pred_proba = model.predict_proba(processed_single_instance)[:, 1][0]
                                pred_label = "CHURN" if pred_proba > 0.5 else "NO CHURN"

                                st.markdown(f"<div class='prediction-result'>Predicted Churn Probability: {pred_proba:.2f} ({pred_label})</div>", unsafe_allow_html=True)
                                if pred_label == "CHURN":
                                    st.warning("This customer profile has a high likelihood of churning! Consider retention strategies.")
                                else:
                                    st.success("This customer profile is likely to remain active. Good job!")
                            except Exception as e:
                                st.error(f"Error making prediction: {e}. Please check your inputs.")
                        else:
                            st.error("Could not process input for prediction. Ensure all necessary features are captured and align with the trained model.")
                else:
                    st.info("No input features configured for What-If analysis. Ensure your dataset has relevant features for user input.")

            else:
                st.warning("Model not trained or features not available for 'What-If' analysis. Please upload a dataset and ensure the model trains successfully.")


# Run the app
if __name__ == "__main__":
    app()

