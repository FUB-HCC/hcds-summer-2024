import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression,LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import fairness_functions as ff
from sklearn.feature_selection import SelectKBest, f_classif
import shap
import dalex as dx
# Set page configuration
st.set_page_config(page_title="Stroke Prediction Dataset Indicators Explorer", layout="wide")

# Custom CSS to increase page width
st.markdown(
    """
    <style>
    .reportview-container .main .block-container{
        max-width: 1200px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Load your dataset
# @st.cache_data
# def load_data():
    
#     return data
data = pd.read_csv("healthcare-dataset-stroke-data.csv")
data = data.dropna()  # Removing rows with NaN values for simplicity
data = data[data['gender'] != 'Other']
data['smoking_status'].replace('formerly smoked', 'smokes', inplace=True)
data = data[data['smoking_status'] != 'Unknown']

# Converting categorical columns to numerical codes
X = data.drop(['id','stroke'], axis=1)
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
# for col in categorical_columns:
#     X[col] = pd.Categorical(X[col]).codes


value_mappings = {}

for col in categorical_columns:
    X[col] = pd.Categorical(X[col])
    value_mappings[col] = dict(enumerate(X[col].cat.categories))
    X[col] = X[col].cat.codes
y = data['stroke']


# Streamlit app
st.title("Stroke Prediction Web App")

# Sidebar content
st.sidebar.header("Dataset Summary")
st.sidebar.subheader("Dataset Description")
st.sidebar.write("""
The dataset provided is aimed at predicting whether a patient is likely to experience a stroke based on a variety of input parameters. 
The dataset contains several attributes related to patient demographics, health conditions, and lifestyle factors.
""")

st.sidebar.subheader("Key Features")
st.sidebar.write("""
- **id**: Type: Integer. Description: Unique identifier for each patient record.
- **gender**: Type: Categorical (Male, Female, Other). Description: The gender of the patient.
- **age**: Type: Numerical. Description: The age of the patient.
- **hypertension**: Type: Binary (0 or 1). Description: Indicates whether the patient has hypertension (1) or not (0).
- **heart_disease**: Type: Binary (0 or 1). Description: Indicates whether the patient has any heart disease (1) or not (0).
- **ever_married**: Type: Categorical (No, Yes). Description: Indicates whether the patient has ever been married.
- **work_type**: Type: Categorical (children, Govt_job, Never_worked, Private, Self-employed). Description: Describes the type of work the patient is engaged in.
- **Residence_type**: Type: Categorical (Rural, Urban). Description: Indicates whether the patient lives in a rural or urban area.
- **avg_glucose_level**: Type: Numerical. Description: The average glucose level in the patient’s blood.
- **bmi**: Type: Numerical. Description: The body mass index of the patient.
- **smoking_status**: Type: Categorical (formerly smoked, never smoked, smokes, Unknown). Description: The smoking status of the patient. "Unknown" indicates that the information is unavailable for this patient.
- **stroke**: Type: Binary (0 or 1). Description: Indicates whether the patient had a stroke (1) or not (0).
""")



tabs = st.tabs(['About the App', 'Web App', 'Dataset', 'Model Performance'])


# # Initialize session state if not already present
# if 'tab_selected' not in st.session_state:
#     st.session_state.tab_selected = 0

# # Function to switch tabs
# def switch_to_tab_1():
#     st.session_state.tab_selected = 1

with tabs[0]:
    st.header("Hello Doctor, Welcome to Our Stroke Prediction App")

    # Create two columns with different widths for image and description
    col1, col2 = st.columns([2, 1])

    with col1:
        st.image('image.jpeg', caption='Welcome to Our Stroke Prediction App', use_column_width=True)

    with col2:
        st.write("""
        This app helps healthcare professionals predict stroke risk using patient health indicators. 
        
        **Benefits:**
        - **Early Detection:** Enables timely interventions for high-risk patients.
        - **Informed Decisions:** Provides data-driven insights for clinical decisions.
        - **Risk Analysis:** Highlights major stroke risk factors.

        **How to Use:**
        1. **Go to the 'Web App' Tab:** Enter patient details.
        2. **Submit Data:** Click the prediction button.
        3. **Review Results:** See stroke risk predictions and suggestions.
        4. **Explore Further:** Use other tabs to delve into data and model performance.
        """)
        # if st.button('Go to Web App'):
        #     switch_to_tab_1()  # Switch to the Web App tab

    # Create a full-width column for the note part below the description
    st.write("---")
    col_note = st.container()
    with col_note:
         st.markdown("""
            <style>
                .header {
                    font-size: 24px;
                    font-weight: bold;
                    color: #2E86C1;
                }
                .subheader {
                    font-size: 20px;
                    font-weight: bold;
                    color: #117A65;
                }
                .important-note {
                    background-color: #F9EBEA;
                    border-left: 6px solid #E74C3C;
                    padding: 10px;
                    margin-bottom: 15px;
                }
                .rely-on {
                    background-color: #D4EFDF;
                    border-left: 6px solid #27AE60;
                    padding: 10px;
                    margin-bottom: 15px;
                }
                .not-rely-on {
                    background-color: #FCF3CF;
                    border-left: 6px solid #F1C40F;
                    padding: 10px;
                }
            </style>
        
            <div class="header">Important Note on Using the Stroke Prediction App</div>
        
            <div class="important-note">
                <b>Important Note:</b> This model is not a substitute for professional medical advice. Use it only as a preliminary screening tool.
            </div>
        
            <div class="subheader">When to Rely on the Predictions:</div>
            <div class="rely-on">
                <ul>
                    <li><b>High-Risk Identification:</b> Effective at identifying patients at high risk of stroke.</li>
                    <li><b>Feature Importance:</b> Highlights key factors contributing to risk.</li>
                </ul>
            </div>
        
            <div class="subheader">When Not to Rely on the Predictions:</div>
            <div class="not-rely-on">
                <ul>
                    <li><b>False Positives:</b> Many flagged as high risk may not actually be at risk.</li>
                    <li><b>Clinical Judgment:</b> Combine with your professional evaluation and diagnostic tests.</li>
                </ul>
            </div>
        
            Click the next tab to proceed to the main app.
        """, unsafe_allow_html=True)

with tabs[2]:
    st.header("Dataset Overview")
    st.subheader("Displaying data:")
    st.dataframe(data)

    data = data.drop('id', axis=1)
    
    # Create a layout with two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Summary statistics of the data:")
        st.write(data.describe())

    with col2:
        st.subheader('Feature Distribution Analysis')
        selected_feature = st.selectbox('Select a feature', list(data.columns))
        def plot_histogram(data, feature):
            fig, ax = plt.subplots()
            ax.hist(data[feature], bins=20, edgecolor='black')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Histogram of {feature}')
            st.pyplot(fig)
        plot_histogram(data, selected_feature)

    # Encode categorical variables
    data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Calculate correlation matrix
    correlation_matrix = data_encoded.corr()

    # Create columns for correlation matrix and heatmap
    col3, col4 = st.columns(2)

    with col3:
        st.subheader('Correlation Matrix')
        st.dataframe(correlation_matrix)

    with col4:
        st.subheader('Correlation Heatmap')
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
        st.pyplot(plt)

    # User selects features to compare
    st.subheader('Feature Relationship Analysis')
    feature = st.selectbox('Select a feature to analyze its relationship with stroke', data.columns)

    # Determine feature types
    num_features = ['age', 'avg_glucose_level', 'bmi']
    binary_cat_features = ['hypertension', 'heart_disease', 'ever_married']
    multi_cat_features = ['gender', 'work_type', 'Residence_type', 'smoking_status']

    if feature in num_features:
        st.subheader(f'Box Plot: {feature} vs Stroke')
        fig, ax = plt.subplots()
        sns.boxplot(x='stroke', y=feature, data=data, palette={'0': "skyblue", '1': "salmon"}, ax=ax)
        ax.set_xlabel('Stroke')
        ax.set_ylabel(feature)
        ax.set_title(f'{feature} vs Stroke')
        st.pyplot(fig)

    elif feature in binary_cat_features:
        st.subheader(f'Bar Plot: {feature} vs Stroke')
        fig, ax = plt.subplots()
        sns.countplot(x=feature, hue='stroke', data=data, ax=ax)
        ax.set_xlabel(feature)
        ax.set_ylabel('Count')
        ax.set_title(f'{feature} vs Stroke')
        ax.legend(title='Stroke', loc='upper right')
        st.pyplot(fig)

    elif feature in multi_cat_features:
        st.subheader(f'Bar Plot: {feature} vs Stroke')
        fig, ax = plt.subplots()
        sns.countplot(x=feature, hue='stroke', data=data, ax=ax)
        ax.set_xlabel(feature)
        ax.set_ylabel('Count')
        ax.set_title(f'{feature} vs Stroke')
        ax.legend(title='Stroke', loc='upper right')
        st.pyplot(fig)

    else:
        st.write("Select a valid feature to analyze its relationship with stroke.")


with tabs[3]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y,random_state=42)

    scaler = StandardScaler()
    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])
    
    # # Feature selection
    # selector = SelectKBest(score_func=f_classif, k=7)
    # X_train_selected = selector.fit_transform(X_train, y_train)
    # X_test_selected = selector.transform(X_test)
    X_test_selected = X_test
    
    smote = SMOTE(random_state=42)
    X_train_resample , y_train_resample  = smote.fit_resample(X_train , y_train)
    logreg_pipeline = LogisticRegression(C=1,penalty='l2',random_state=42)

    logreg_pipeline.fit(X_train_resample, y_train_resample)

    # Predictions
    y_pred = logreg_pipeline.predict(X_test_selected)
    y_prob = logreg_pipeline.predict_proba(X_test_selected)[:, 1]
    threshold = 0.35# Example threshold, you may need to adjust based on precision-recall curve
    y_pred_adjusted = (y_prob >= threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred_adjusted)
    conf_matrix = confusion_matrix(y_test, y_pred_adjusted)
    class_report = classification_report(y_test, y_pred_adjusted)
    roc_auc = roc_auc_score(y_test, y_pred_adjusted)
    fpr, tpr, _ = roc_curve(y_test, y_pred_adjusted)
    X_test_with_predictions = pd.DataFrame(X_test, columns=X_test.columns)
    X_test_with_predictions['y_pred'] = y_pred_adjusted
    X_test_with_predictions['stroke'] = y_test

    st.header("Model Performance")
    col3, col4 = st.columns(2)
    with col3:
        st.subheader('Fairness Score')
        # The following function returns the probability of a given group (arg3) to be assigned to the positively predicted class (arg5). 
        prob_female = ff.group_fairness(X_test_with_predictions, "gender", 0, "y_pred", 1)
        prob_male = ff.group_fairness(X_test_with_predictions, "gender", 1, "y_pred", 1)
        st.write("Probability of females being assigned to the positive class:", prob_female)
        st.write("Probability of males being assigned to the positive class:", prob_male)

        prob_female = ff.conditional_statistical_parity(X_test_with_predictions, "gender", 0, "y_pred", 1, "heart_disease", 1)
        prob_male = ff.conditional_statistical_parity(X_test_with_predictions, "gender", 1, "y_pred", 1,"heart_disease", 1)
        st.write("Probability of females being assigned to the positive class with condition(heart_disease):", prob_female)
        st.write("Probability of males being assigned to the positive class with condition(heart_disease):", prob_male)

        prob_female = ff.predictive_parity(X_test_with_predictions, "gender", 0, "y_pred", "stroke")
        prob_male = ff.predictive_parity(X_test_with_predictions, "gender", 1, "y_pred", "stroke")
        st.write("Probability that females predicted to have stroke actually have stroke:", prob_female)
        st.write("Probability that males predicted to have stroke actually have stroke:", prob_male)

        prob_female = ff.fp_error_rate_balance(X_test_with_predictions, "gender", 0, "y_pred", "stroke")
        prob_male = ff.fp_error_rate_balance(X_test_with_predictions, "gender", 1, "y_pred", "stroke")
        st.write("Probability that females predicted to have stroke actually not have stocke:", prob_female)
        st.write("Probability that males predicted to have stroke actually not have stocke:", prob_male)

    with col4:
        st.subheader("Accuracy")
        st.write(accuracy)
        st.subheader("Classification Report")
        st.text(class_report)
        st.subheader("ROC AUC Score")
        st.write(roc_auc)
        

    #confusion matrix
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',linewidths=1, linecolor='black', ax=ax,annot_kws={'size': 15, 'weight': 'bold', 'color': 'grey'})
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        st.pyplot(fig)
        # Explanation of the confusion matrix
        st.write("""
                             
        The left part of the image shows a confusion matrix for a binary classification problem. The matrix is a 2x2 grid representing the counts of true positive, true negative, false positive, and false negative predictions made by the model.
        Here’s what each cell in the matrix represents:
                 
        - **True Negative (TN)**: The top-left cell shows 420 instances where the model correctly predicted the negative class (0)
        - **False Positive (FP)**: The top-right cell shows 229 instances where the model incorrectly predicted the positive class (1) for actual negative class instances (0)
        - **False Negative (FN)**: The bottom-left cell shows 9 instances where the model incorrectly predicted the negative class (0) for actual positive class instances (1).
        - **True Positive (TP)**: The bottom-right cell shows 27 instances where the model correctly predicted the positive class (1).
       """)
        #ROC curve
    with col6:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc="lower right")
        st.pyplot(fig)
        # Explanation of the ROC curve
        st.write("""
                 
        The right part of the image shows an ROC (Receiver Operating Characteristic) curve, which is a graphical representation of the model's diagnostic ability. 
        Here are the details:

                 
        - **X-axis (False Positive Rate)**: Represents the proportion of actual negatives that were incorrectly classified as positives (FP / (FP + TN)).
        - **Y-axis (True Positive Rate)**: Represents the proportion of actual positives that were correctly classified (TP / (TP + FN)).
        - **ROC Curve**: The orange line plots the trade-off between the true positive rate and false positive rate at various threshold settings.
        - **Diagonal Line**: The dashed blue line represents a random classifier's performance, where the true positive rate equals the false positive rate.
        - **AUC (Area Under the Curve)**: The area under the ROC curve is 0.70, indicating the model has moderate discriminative ability.    
      """)

  
with tabs[1]:
    st.header("Input Patient Data")

    # Create columns for input fields
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", options=list(value_mappings['gender'].keys()), format_func=lambda x: value_mappings['gender'][x])
        age = st.number_input("Age", min_value=10, max_value=120, step=1, value=30)
        hypertension = st.selectbox("Hypertension", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        heart_disease = st.selectbox("Heart Disease", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        ever_married = st.selectbox("Ever Married", options=list(value_mappings['ever_married'].keys()), format_func=lambda x: value_mappings['ever_married'][x])

    with col2:
        work_type = st.selectbox("Work Type", options=list(value_mappings['work_type'].keys()), format_func=lambda x: value_mappings['work_type'][x])
        residence_type = st.selectbox("Residence Type", options=list(value_mappings['Residence_type'].keys()), format_func=lambda x: value_mappings['Residence_type'][x])
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=300.0, step=0.1, value=100.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, step=0.1, value=25.0)
        smoking_status = st.selectbox("Smoking Status", options=list(value_mappings['smoking_status'].keys()), format_func=lambda x: value_mappings['smoking_status'][x])

    # Predict stroke risk
    patient_data = np.array([[gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status]])
    patient_data[:, [1, 7, 8]] = scaler.transform(patient_data[:, [1, 7, 8]])
    y_prob = logreg_pipeline.predict_proba(patient_data)[:, 1]
    threshold = 0.35  # Example threshold, you may need to adjust based on precision-recall curve
    y_pred_adjusted = (y_prob >= threshold).astype(int)

    # Display prediction
    st.header("Prediction Result")
    if y_pred_adjusted[0] == 1:
        st.markdown("""
        <div style="
            border: 2px solid red; 
            border-radius: 10px; 
            background-color: #ffcccc;
            padding: 10px; 
            color: red;
            font-size: 20px;
            font-weight: bold;">
            The patient is likely to experience a stroke.<br>
            <span style="font-weight: normal;">Suggestion: Please send this patient for further medical tests necessary for stroke diagnosis and prevention.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            border: 2px solid green; 
            border-radius: 10px; 
            background-color: #ccffcc;
            padding: 10px; 
            color: green;
            font-size: 20px;
            font-weight: bold;">
            The patient is unlikely to experience a stroke.<br>
            <span style="font-weight: normal;">Suggestion: Advise to continue regular check-ups to maintain good health.</span>
        </div>
        """, unsafe_allow_html=True)

    st.header("Insight of the Prediction")

    # Feature Importance using Dalex
    explainer = dx.Explainer(logreg_pipeline, X_train_resample, y_train_resample, label="Feature contribution")
    shap_values = explainer.predict_parts(patient_data)

    st.write("#### Feature Importance (SHAP Values)")

    shap_summary = shap_values.result[['variable_name', 'contribution']].iloc[1:-1].copy()
    shap_summary = shap_summary.sort_values(by='contribution', ascending=False)

    # Create columns for plot and explanation
    plot_col, explanation_col = st.columns(2)

    with plot_col:
        plt.figure(figsize=(10, 6))
        plt.grid()
        bars = plt.barh(shap_summary['variable_name'], shap_summary['contribution'], color=['blue' if c >= 0 else 'red' for c in shap_summary['contribution']])
        st.pyplot(plt)

        st.markdown(f"""
        The plot helps to understand which features are driving the model's prediction and to what extent. It provides a transparent view of the model's decision-making process.
        
        In the plot:
        - **Magnitude:** The size of the bar indicates the strength of the feature's influence on the prediction. Larger bars have a more significant impact.
        - **Direction:** The color coding (and sign) of the bars indicates whether the influence is positive (blue) or negative (red).
        """)

    def generate_explanation(shap_values):
        # Sort SHAP values by absolute contribution (excluding the intercept)
        shap_sorted = shap_values.result.iloc[:-1][shap_values.result['variable_name'] != 'intercept'].copy()
        shap_sorted['abs_contribution'] = shap_sorted['contribution'].abs()
        shap_sorted = shap_sorted.sort_values(by='abs_contribution', ascending=False).head(3)

        most_contributing_features = shap_sorted.variable_name.values
        explanation = []

        for index, row in shap_values.result.iterrows():
            if index == len(shap_values.result):
                pass
            if row['variable'] == 'intercept':
                pass
            else:
                feature = row['variable_name']
                if feature != '':
                    contribution = row['contribution']
                    if contribution > 0:
                        explanation.append(f"**{feature}** increases the prediction by {contribution:.4f}.")
                    else:
                        explanation.append(f"**{feature}** decreases the prediction by {contribution:.4f}.")

        return "\n\n".join(explanation), most_contributing_features

    textual_explanation, imp_feature = generate_explanation(shap_values)

    with explanation_col:

        st.markdown(f"""
        <div style='border: 1px solid #ccc; padding: 10px;'>
            <h4>Interpretation of the plot:</h4>
            <p>{textual_explanation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display top influential features in a separate box
        st.markdown(f"""
        <div style='border: 1px solid #ccc; padding: 10px;'>
            <h4>Most influential features:</h4>
            <ul>
                <li>{imp_feature[0]}</li>
                <li>{imp_feature[1]}</li>
                <li>{imp_feature[2]}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
