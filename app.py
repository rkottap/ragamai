
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_absolute_error, mean_squared_error,
    roc_curve, auc, log_loss, r2_score
)
import warnings
warnings.filterwarnings('ignore')

# Load models (original + SMOTE)
rf_orig = joblib.load('rf_model_original.pkl')
xgb_orig = joblib.load('xgb_model_original.pkl')
knn_orig = joblib.load('knn_model_original.pkl')
svm_orig = joblib.load('svm_model_original.pkl')
rf_smote = joblib.load('rf_model_smote.pkl')
xgb_smote = joblib.load('xgb_model_smote.pkl')
knn_smote = joblib.load('knn_model_smote.pkl')
svm_smote = joblib.load('svm_model_smote.pkl')

# Load dataset
df = pd.read_csv("California Hospital Inpatient Mortality Rates and Quality Ratings .csv")

# Initial cleaning and preprocessing steps
df = df[df['HOSPITAL'] != 'STATEWIDE']  
df.replace({'.': np.nan, ' ': np.nan, '': np.nan}, inplace=True)  
df = df[df['Hospital Ratings'].notna()]  
df.drop(columns=["LONGITUDE", "LATITUDE", "OSHPDID"], errors="ignore", inplace=True)

# Data imputation
df["Risk Adjuested Mortality Rate"] = pd.to_numeric(df["Risk Adjuested Mortality Rate"], errors="coerce")
df['Risk Adjuested Mortality Rate'] = df.groupby('HOSPITAL')['Risk Adjuested Mortality Rate'].transform(lambda x: x.fillna(x.median()))
df["# of Deaths"] = pd.to_numeric(df["# of Deaths"], errors="coerce")
df["# of Cases"] = pd.to_numeric(df["# of Cases"], errors="coerce")
df["HOSPITAL"].fillna("Unknown", inplace=True)
df["COUNTY"].fillna("Unknown", inplace=True)
df["Procedure/Condition"] = df.groupby("HOSPITAL")["Procedure/Condition"].transform(lambda x: x.fillna(x.mode()[0]))

# Outliers removal function
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

df = remove_outliers(df, ["Risk Adjuested Mortality Rate", "# of Deaths", "# of Cases"])

# Feature engineering
df['Mortality_Per_Case'] = df['Risk Adjuested Mortality Rate'] / df['# of Cases']
df["Mortality_Per_Case"] = pd.to_numeric(df["Mortality_Per_Case"], errors="coerce")

# Label Encoding function

le_proc = LabelEncoder()
le_hosp = LabelEncoder()
le_county = LabelEncoder()
le_rating = LabelEncoder()

df["Procedure/Condition"] = le_proc.fit_transform(df["Procedure/Condition"])
df["HOSPITAL"] = le_hosp.fit_transform(df["HOSPITAL"])
df["COUNTY"] = le_county.fit_transform(df["COUNTY"])

custom_label_map = {'Worse': 0, 'As Expected': 1, 'Better': 2}
df["Hospital Ratings"] = df["Hospital Ratings"].map(custom_label_map)
label_mapping = custom_label_map

# Split the dataset into training and testing sets
X = df.drop(columns=["Hospital Ratings"])
y = df["Hospital Ratings"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Hardcoded metrics for original models
metrics_orig = [
    {
        "Model": "Random Forest (Original)",
        "Accuracy": 0.987322,
        "Precision": 0.979566,
        "Recall": 0.869022,
        "F1 Score": 0.917880,
        "MAE": 0.012678,
        "MSE": 0.012678,
        "RMSE": 0.112595,
        "MAPE": 0.094386,
        "ROC-AUC": 0.996863,
        "Log Loss": 0.048855,
        "R-squared": 0.787531
    },
    {
        "Model": "XGBoost (Original)",
        "Accuracy": 0.991091,
        "Precision": 0.968930,
        "Recall": 0.932568,
        "F1 Score": 0.952020,
        "MAE": 0.008090,
        "MSE": 0.008090,
        "RMSE": 0.094386,
        "MAPE": 0.008090,
        "ROC-AUC": 0.998762,
        "Log Loss": 0.027313,
        "R-squared": 0.850698
    },
    {
        "Model": "KNN (Original)",
        "Accuracy": 0.939866,
        "Precision": 0.617511,
        "Recall": 0.411663,
        "F1 Score": 0.450745,
        "MAE": 0.060476,
        "MSE": 0.061162,
        "RMSE": 0.247390,
        "MAPE": "inf",
        "ROC-AUC": 0.865517,
        "Log Loss": 0.495861,
        "R-squared": -0.025017
    },
    {
        "Model": "SVM (Original)",
        "Accuracy": 0.940084,
        "Precision": 0.313346,
        "Recall": 0.333333,
        "F1 Score": 0.323031,
        "MAE": 0.014717,
        "MSE": 0.013334,
        "RMSE": 0.115559,
        "MAPE": 0.115559,
        "ROC-AUC": 0.540120,
        "Log Loss": 1.162723,
        "R-squared": -0.004919
    }
]

# Hardcoded metrics for SMOTE models
metrics_smote = [
    {
        "Model": "Random Forest (SMOTE)",
        "Accuracy": 0.986466,
        "Precision": 0.921968,
        "Recall": 0.952807,
        "F1 Score": 0.935917,
        "MAE": 0.013534,
        "MSE": 0.013534,
        "RMSE": 0.116337,
        "MAPE": "inf",
        "ROC-AUC": 0.993869,
        "Log Loss": 0.086729,
        "R-squared": 0.773175
    },
    {
        "Model": "XGBoost (SMOTE)",
        "Accuracy": 0.975158,
        "Precision": 0.838795,
        "Recall": 0.931741,
        "F1 Score": 0.874964,
        "MAE": 0.025013,
        "MSE": 0.025053,
        "RMSE": 0.159234,
        "MAPE": "inf",
        "ROC-AUC": 0.992764,
        "Log Loss": 0.080711,
        "R-squared": 0.850698
    },
    {
        "Model": "KNN (SMOTE)",
        "Accuracy": 0.868083,
        "Precision": 0.516060,
        "Recall": 0.830767,
        "F1 Score": 0.585802,
        "MAE": 0.132945,
        "MSE": 0.132945,
        "RMSE": 0.367425,
        "MAPE": "inf",
        "ROC-AUC": 0.899372,
        "Log Loss": 1.266048,
        "R-squared": 0.575603
    },
    {
        "Model": "SVM (SMOTE)",
        "Accuracy": 0.617612,
        "Precision": 0.398739,
        "Recall": 0.699808,
        "F1 Score": 0.375301,
        "MAE": 0.388898,
        "MSE": 0.401919,
        "RMSE": 0.633971,
        "MAPE": "inf",
        "ROC-AUC": 0.844654,
        "Log Loss": 0.847338,
        "R-squared": -5.735828
    }
]

# Convert to DataFrame for easier table display
metrics_all = pd.DataFrame(metrics_orig + metrics_smote)


# Visualization Functions

# Function to plot missing data heatmap

def plot_feature_distribution():

    st.subheader("Feature Distribution")   
    fig, ax = plt.subplots()
    sns.histplot(df['Risk Adjuested Mortality Rate'], kde=True, ax=ax)
    st.pyplot(fig)

def plot_correlation_matrix():
    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots()
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

def plot_target_distribution():
    st.subheader("Hospital Ratings Distribution")
 
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Hospital Ratings", ax=ax)
    st.pyplot(fig)

def plot_boxplot_by_rating():
    st.subheader("Distribution of Cases by Hospital Ratings")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(x=df["Hospital Ratings"], y=df["# of Cases"], palette="Blues_d", ax=ax)
    ax.set_title("Distribution of Cases by Hospital Ratings")
    ax.set_xlabel("Hospital Ratings")
    ax.set_ylabel("Number of Cases")
    st.pyplot(fig)

def plot_top_counties():
    st.subheader("Top 15 Counties by Case Count")

    top_counties = df.groupby("COUNTY")["# of Cases"].sum().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_counties.values, y=top_counties.index, palette="viridis", ax=ax)
    ax.set_title("Top 15 Counties by Case Count")
    ax.set_xlabel("Total Cases")
    ax.set_ylabel("County")
    st.pyplot(fig)

def plot_top_hospitals():
    st.subheader("Top 10 Hospitals by Case Count")
    
    top_hospitals = df.groupby("HOSPITAL")["# of Cases"].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))    
    sns.barplot(x=top_hospitals.values, y=top_hospitals.index, palette="mako", ax=ax)
    ax.set_title("Top 10 Hospitals by Case Count")
    ax.set_xlabel("Total Cases")
    ax.set_ylabel("Hospital")
    st.pyplot(fig)


# Main Page Functions
def home_page():
    st.title("Welcome to the Hospital Ratings Prediction App!")
    st.markdown("""This app allows you to:
    - **Explore the dataset** with Exploratory Data Analysis (EDA).
    - **Evaluate machine learning models** like Random Forest, XGBoost, KNN, and SVM.
    - **Compare model metrics** for better decision-making.
    """)
    if st.button("Go to EDA"):
        st.session_state.page = "EDA"
    elif st.button("Go to Model Evaluation"):
        st.session_state.page = "Model Evaluation"
    elif st.button("Go to Model Comparison Metrics"):
        st.session_state.page = "Model Comparison Metrics"

# ---- EDA Page ----
def eda_page():
   
    if st.button("Back to Home"):
        st.session_state.page = "Home"

    st.title("Exploratory Data Analysis (EDA)")

    plot_feature_distribution()
    plot_correlation_matrix()
    plot_target_distribution()
    plot_boxplot_by_rating()
    plot_top_counties()
    plot_top_hospitals()

# ---- Model Evaluation Page ----
def model_evaluation_page():
    
    if st.button("Back to Home"):
        st.session_state.page = "Home"

    st.title("Model Evaluation")
    
    model_choice = st.selectbox("Select a Model", [
        "Random Forest (Original)", "XGBoost (Original)", "KNN (Original)", "SVM (Original)",
        "Random Forest (SMOTE)", "XGBoost (SMOTE)", "KNN (SMOTE)", "SVM (SMOTE)"
    ])

    model = None
    if model_choice == "Random Forest (Original)":
        model = rf_orig
    elif model_choice == "XGBoost (Original)":
        model = xgb_orig
    elif model_choice == "KNN (Original)":
        model = knn_orig
    elif model_choice == "SVM (Original)":
        model = svm_orig
    elif model_choice == "Random Forest (SMOTE)":
        model = rf_smote
    elif model_choice == "XGBoost (SMOTE)":
        model = xgb_smote
    elif model_choice == "KNN (SMOTE)":
        model = knn_smote
    elif model_choice == "SVM (SMOTE)":
        model = svm_smote

    evaluate_model(model, X_test, y_test, model_choice)

# ---- Model Comparison Metrics Page ----
def model_comparison_page():

    if st.button("Back to Home"):
        st.session_state.page = "Home"

    st.title("Model Comparison Metrics")
    st.write(metrics_all)

# Visualization for Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    
    # Create a figure object
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Plot the confusion matrix using seaborn heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
                xticklabels=["Worse", "As Expected", "Better"], 
                yticklabels=["Worse", "As Expected", "Better"], ax=ax)
    
    # Set labels and title
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    
    # Display the plot in the Streamlit app
    st.pyplot(fig)

# Visualization for ROC Curve
def plot_roc_curve(y_true, y_prob, title="ROC-AUC Curve"):
    # Extract probabilities for class '1' ("As Expected")
    y_prob_class_1 = y_prob[:, 1]  # Class 1: "As Expected"

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob_class_1, pos_label=1)  # pos_label=1 for "As Expected"
    roc_auc = auc(fpr, tpr)

    # Create a figure object
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="b", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")

    # Display the plot in the Streamlit app
    st.pyplot(fig)


# ---- Model Evaluation Function ----
def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    st.write(f"Evaluating {model_name}...")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    st.write(f"Precision: {precision_score(y_test, y_pred, average='macro')}")
    st.write(f"Recall: {recall_score(y_test, y_pred, average='macro')}")
    st.write(f"F1 Score : {f1_score(y_test, y_pred,average='macro')}")
    if y_prob is not None:
        st.write(f"ROC-AUC: {roc_auc_score(y_test, y_prob, multi_class='ovo', average='macro')}")
        st.write(f"Log Loss: {log_loss(y_test, y_prob)}")
    plot_confusion_matrix(y_test, y_pred)
    if y_prob is not None:
        plot_roc_curve(y_test, y_prob)

# Main Logic
if "page" not in st.session_state:
    st.session_state.page = "Home"

if st.session_state.page == "Home":
    home_page()
elif st.session_state.page == "EDA":
    eda_page()
elif st.session_state.page == "Model Evaluation":
    model_evaluation_page()
elif st.session_state.page == "Model Comparison Metrics":
    model_comparison_page()
