import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import fairness_functions as ff
from make_visualizations import plot_confusion_matrix, plot_roc
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

DATA_PATH = "dataset_enriched.csv"

@st.cache_data
def app_init():
    title = '<h1 style="font-family:sans-serif; color:RoyalBlue; font-size: 26px; padding-bottom: 0;">About the model</h1>'
    st.markdown(title, unsafe_allow_html=True)
    st.write("---")
    st.markdown("""
        A **Support Vector Machine (SVM)** with a straightforward approach was used for model development. To address the imbalance in the
        data, techniques (SMOTE etc.) were used to ensure the model accurately identifies asthma even when cases are rare. 
        Additionally, we standardized the data to ensure all patient information is on the same scale.
    """)
    st.markdown('<h1 style="font-family:sans-serif; color:RoyalBlue; font-size: 26px;"> Model Performance Evaluation</h1>', unsafe_allow_html=True)
    
    data = pd.read_csv(DATA_PATH)
    y_true = data['Diagnosis']
    y_pred = data['Prediction']
    
    cm = confusion_matrix(y_true, y_pred)
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    class_report = classification_report(y_true, y_pred, output_dict=True)
    dict_to_show = {label: {subkey: class_report[key][subkey] for subkey in ['precision', 'recall', 'f1-score']} for label, key in zip(['No Asthma', 'Asthma'], ['0', '1'])}
   
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"Accuracy: {class_report['accuracy']: .2f}")
    with col2:
        st.dataframe(dict_to_show)
    
    col3, col4 = st.columns(2)
    with col3:
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(plt.gcf())
    with col4:
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        st.pyplot(plt.gcf())
    
    st.markdown('<h1 style="font-family:sans-serif; color:RoyalBlue; font-size: 26px;"> Interpretation</h1>', unsafe_allow_html=True)
    st.markdown("""
    - **Accuracy**: Measures how often the model is correct. High accuracy indicates the model performs well.
    - **Precision**: Indicates how many of the positive predictions are actually correct.
    - **Recall**: Indicates how many of the actual positives the model correctly identified.
    - **F1-Score**: Harmonic mean of precision and recall. High F1-score indicates a balance between precision and recall.
    - **Confusion Matrix**: Shows the counts of true positive, true negative, false positive, and false negative predictions.
    - **ROC Curve**: Graphical representation of the true positive rate vs. the false positive rate.
    - **AUC (Area Under the Curve)**: Measures the overall ability of the model to discriminate between positive and negative classes. Higher AUC indicates better performance.
    """)
    st.markdown('<h1 style="font-family:sans-serif; color:RoyalBlue; font-size: 26px;"> Global Explanations</h1>', unsafe_allow_html=True)
    
    model = joblib.load('SVM_linear_scaled_p.sav')
    coef_df = pd.DataFrame({
        'Feature': ['Age', 'Gender', 'Ethnicity', 'Education Level', 'BMI', 'Smoking', 'Physical Activity',
                    'Diet Quality', 'Sleep Quality', 'Pollution Exposure', 'Pollen Exposure', 'Dust Exposure',
                    'Pet Allergy', 'Family History of Asthma', 'History of Allergies', 'Eczema', 'Hay Fever',
                    'Gastroesophageal Reflux', 'Lung Function (FEV1)', 'Lung Function (FVC)', 'Wheezing',
                    'Shortness of Breath', 'Chest Tightness', 'Coughing', 'Nighttime Symptoms', 'Exercise Induced'],
        'Weight': model.coef_[0]
    }).sort_values(by='Weight', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Weight', y='Feature', data=coef_df, palette='viridis')
    plt.title('Feature Weights')
    plt.xlabel('Weight')
    plt.ylabel('Feature')

    for i, (value, name) in enumerate(zip(coef_df['Weight'], coef_df['Feature'])):
        plt.text(value, i, f'{value:.2f}', va='center', ha='right' if value < 0 else 'left', color='black')

    plt.gca().xaxis.set_visible(False)
    plt.grid(False)

    st.pyplot(plt.gcf())

    st.markdown("""
    The plot above shows the weights of each feature in the SVM model. Features with higher absolute values are more influential in the prediction. Positive weights push the prediction towards a positive diagnosis (asthma), while negative weights push the prediction towards a negative diagnosis (no asthma). This helps to understand which features are contributing the most to the model's decision.
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 style="font-family:sans-serif; color:RoyalBlue; font-size: 26px;"> Fairness Evaluation</h1>', unsafe_allow_html=True)
    st.markdown("""
    **Protected Variables:** Age, gender, ethnicity, education level, Family History

    **Fairness Metrics:** Evaluation of model performance across different demographic groups to ensure
                 fairness and avoid bias with respect to the following fairness metrics:
                
    -  The **Group Fairness** measures the difference in positive prediction
        rates between different groups; e.g.: Is it equally likely for male and female to be predicted
        asthma? A smaller difference indicates fairer treatment across groups.
    
    -   The **Conditional Statistical Parity** measures, similar to the group fairness,
                     the difference in positive prediction rates between different groups
                (e.g., male vs. female) within another group, like in our case among all smokers.
                    A smaller difference indicates fairer treatment across groups.
                
    -  The **Predictive Parity** measures the positive predicted value of
                    different groups; e.g. is it equally likely for male and female when they are predicted
                    with asthma to actually be diagnosed asthma?
                    A smaller difference indicates fairer treatment across groups. Higher values, in general,
                    mean better reliability for the model.
                
    -  The **False Positive Error Rate Balance** measures the False 
                    Positive Rate for different groups; e.g., is it equally likely for male and female 
                    that are actually not diagnosed asthma to be falsely predicted with asthma?
                    A smaller difference indicates fairer treatment across groups.
                    In general, a low value means higher reliability for the model.
    """)

def explore_fairness():
    df = pd.read_csv(DATA_PATH)

    FF = ['Group Fairness', 'Conditional Statistical Parity', 'Predictive Parity', 'False Positive Error Rate Balance']
    fairnes_metrics = st.selectbox("Select fairness metrics:", FF)

    protected_attribute = st.selectbox("Select a protected attribute for analysis:", ['Gender', 'Age', 'Ethnicity','EducationLevel', 'FamilyHistoryAsthma'])
    positive_class_name = st.selectbox("Select whether No Asthma or Asthma should be considered as positive class.", ['No Asthma', 'Asthma'])
    target_class_mapper = {'No Asthma': 0, 'Asthma': 1}
    positive_class = target_class_mapper[positive_class_name]
    
    gender_map = {0: "Male", 1: "Female"}
    ethnicity_map = {0: "Caucasian", 1: "African American", 2: "Asian", 3: "Other"}
    education_level_map = {0: "None", 1: "High School", 2: "Bachelor's", 3: "Higher"}
    family_history_map = {0: "No", 1: "Yes"}

    def map_values(attribute, value):
        if attribute == 'Gender':
            return gender_map[value]
        elif attribute == 'Ethnicity':
            return ethnicity_map[value]
        elif attribute == 'EducationLevel':
            return education_level_map[value]
        elif attribute == 'FamilyHistoryAsthma':
            return family_history_map[value]
        else:
            return value

    if fairnes_metrics == 'Group Fairness':
        fairness_dict = {}
        for value in df[protected_attribute].unique():
            mapped_value = map_values(protected_attribute, value)
            fairness = ff.group_fairness(df, protected_attribute, value, "Prediction", positive_class)
            fairness_dict[mapped_value] = fairness
    elif fairnes_metrics == 'Conditional Statistical Parity':
        fairness_dict = {}
        for value in df[protected_attribute].unique():
            mapped_value = map_values(protected_attribute, value)
            fairness = ff.conditional_statistical_parity(df, protected_attribute, value, "Prediction", positive_class, 'Smoking', 1)
            fairness_dict[mapped_value] = fairness
    elif fairnes_metrics == 'Predictive Parity':
        fairness_dict = {}
        for value in df[protected_attribute].unique():
            mapped_value = map_values(protected_attribute, value)
            fairness = ff.predictive_parity(df, protected_attribute, value, "Prediction", "Diagnosis")
            fairness_dict[mapped_value] = fairness
    elif fairnes_metrics == 'False Positive Error Rate Balance':
        fairness_dict = {}
        for value in df[protected_attribute].unique():
            mapped_value = map_values(protected_attribute, value)
            fairness = ff.fp_error_rate_balance(df, protected_attribute, value, "Prediction", "Diagnosis")
            fairness_dict[mapped_value] = fairness
    fig = go.Figure(data=[go.Bar(x=list(fairness_dict.keys()), y=list(fairness_dict.values()))])
    fig.update_layout(
        title=f'{fairnes_metrics} for protected variable {protected_attribute}',
        xaxis_title='Group Value',
        yaxis_title=f'{fairnes_metrics}'
    )
    st.plotly_chart(fig)


if __name__ == '__main__':
    data = pd.read_csv(DATA_PATH)
    app_init()
    explore_fairness()
