# Customer Churn Prediction & Analysis App

## 📊 Project Overview

This interactive web application, built with **Streamlit**, provides a comprehensive platform for analyzing customer churn patterns, identifying key influencing factors, and making predictions. Inspired by the IBM Telco Churn Dataset, this project aims to offer actionable insights to businesses looking to reduce customer attrition and enhance retention strategies.

The app is designed with a clean, intuitive user interface and a dark theme for an engaging user experience, making complex data analysis accessible.

## ✨ Key Features

* **Dynamic Data Upload:** Easily upload your customer churn data in CSV or Excel format.
* **Interactive Exploratory Data Analysis (EDA):**
    * Overview of dataset statistics (dimensions, missing values, data types).
    * Visualizations of churn distribution (pie chart).
    * Interactive bar charts and box plots to explore categorical and numerical feature distributions against churn, with a consistent color scheme (Green for No Churn, Red for Churn).
* **Feature Importance Insights:**
    * Trains a RandomForestClassifier model on the uploaded data.
    * Displays the top 10 most influential features driving customer churn using a clear bar chart.
* **Deep Dive Analysis:**
    * Visualizes top churn reasons (if available in the dataset).
    * Analyzes churn rates by tenure groups and contract types.
* **Model Predictions & Performance:**
    * Shows the accuracy score of the trained churn prediction model.
    * Allows users to download a `predictions.csv` file containing the original data augmented with predicted churn values and probabilities.
* **"What-If" Scenario Analysis:** A unique feature allowing users to manually adjust various customer attributes (e.g., tenure, monthly charges, internet service) and instantly see the predicted churn probability for that hypothetical customer profile. This demonstrates the model's practical utility and interpretability.

## 🛠️ Technologies Used

* **Python:** The core programming language.
* **Streamlit:** For building the interactive web application and user interface.
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Scikit-learn:** For machine learning model (RandomForestClassifier) training and evaluation.
* **Plotly Express:** For creating attractive and interactive data visualizations.
* **Matplotlib:** (Used internally by Plotly/Streamlit for some plotting functionalities).

## 🚀 How to Run Locally

To run this application on your local machine, follow these steps:

1.  **Clone the repository (or download `app.py` and `requirements.txt`):**
    ```bash
    git clone [https://github.com/](https://github.com/)[Your GitHub Username]/customer-churn-app.git
    cd customer-churn-app
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    Your app will open in your default web browser at `http://localhost:8501`.

## ☁️ Deployment

This application is deployed on **Streamlit Community Cloud** and can be accessed directly via the following URL:

**[Deployed App URL Here]**
*(https://janani-customer-churn-app.streamlit.app/)*p

## 📝 Usage

1.  **Upload Data:** On the sidebar, use the "Choose a CSV or Excel file" button to upload your customer churn dataset. Ensure your dataset contains a 'Churn Value' column (0 for No Churn, 1 for Churn) for full functionality.
2.  **Navigate Tabs:** Explore the different tabs to gain insights:
    * **Overview & EDA:** Get a summary of your data and interactive exploratory visualizations.
    * **Feature Importance:** Understand which factors are most critical in predicting churn.
    * **Deep Dive Analysis:** Discover specific churn reasons and trends by tenure and contract type.
    * **Predictions:** View the model's accuracy and download a CSV with predictions for your entire dataset.
    * **What-If Analysis:** Experiment with hypothetical customer profiles to see their predicted churn probability.

## 📚 Dataset Information

This project is inspired by the **IBM Telco Churn Dataset** (available on Kaggle). The dataset typically includes columns such as:

* `CustomerID`
* Demographic information (`Gender`, `Senior Citizen`, `Partner`, `Dependents`)
* Service information (`Phone Service`, `Multiple Lines`, `Internet Service`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies`)
* Contract and billing details (`Contract`, `Paperless Billing`, `Payment Method`, `Monthly Charges`, `Total Charges`, `Tenure Months`)
* Churn-related metrics (`Churn Label`, `Churn Value`, `Churn Score`, `CLTV`, `Churn Reason`)

## 🤝 Contact & Credits

* **Developed by:** [Janani D]
* **GitHub:** https://share.google/lnCybexc45sAO5mlu
* **LinkedIn:** https://www.linkedin.com/in/janani-d-157204361/

---
