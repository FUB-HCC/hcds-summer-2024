import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np


global model_retrained, model_original, X_train_original, X_test_original, y_train_original, y_test_original, y_pred_original, X_train_retrained, X_test_retrained, y_train_retrained, y_test_retrained, y_pred_retrained, features_to_drop

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('NPHA-doctor-visits.csv')
    return data

# Function to train a model
def train_model(data, features_to_drop):
    if features_to_drop != []:
        st.write('### <span style="color: #007BFF;">Model retrained successfully!</span>', unsafe_allow_html=True)
        
    X = data.drop(['Number of Doctors Visited'] + features_to_drop, axis=1)
    y = data['Number of Doctors Visited']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    return model, X_train, X_test, y_train, y_test, y_pred

    
def render_dataframe(df):
    df_html = df.to_html()
    return f'<div class="center-table">{df_html}</div>'


def show_classification_report(model, y_test, y_pred):
        
    with st.expander("Classification Report"):
        report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        st.markdown(render_dataframe(report), unsafe_allow_html=True)
        
        st.write("""
        ### Metric Explanations (for Class 1 (type: visits 0-1 doctor), 2 (type: visits 2-3 doctors) and 3 (type: visits 4 or more doctors ) ) :

        **Precision**: Measures how many of the predicted positive cases are actually correct.
        - For example, for class 1, if precision is 0.176, it means 17.6% of the cases predicted as class 1 are actually class 1.

        **Recall (Sensitivity)**: Measures how many of the actual positive cases the model predicted correctly.
        - For example, for class 1, if recall is 0.079, it means the model correctly identified 7.9% of all actual class 1 cases.

        **F1-score**: It is is a metric that balances both Precision and Recall. It provides a single value that indicates how well a model balances between accurately predicting positive instances (Precision) and capturing all positive instances (Recall). A higher F1-score indicates better performance, with 1 being the best possible score and 0 the worst.
        
        **Support**: Support indicates the number of instances (samples) for each class in the dataset used to evaluate the model's performance. It provides insight into the distribution of classes and how many instances belong to each class.
        

        **Accuracy**: Measures the overall accuracy of the model across all classes.
        - If accuracy is 0.5 (or 50%), it means the model correctly predicts 50% of all cases.

        **Macro Average**: It is the average of the precision, recall, and F1-score across all classes, treating all classes equally.

        **Weighted Average**: Average of the precision, recall, and F1-score, weighted by the number of instances of each class. This metric gives more weight to classes with more samples.
        """)

@st.cache_data(hash_funcs={RandomForestClassifier: lambda _: None})
def explain_lime_and_shap(X_train, model, y_train):
    
        
    
    # LIME 
    
    explainer = LimeTabularExplainer(X_train.values, mode="classification", training_labels=y_train, feature_names=X_train.columns)
    sample_idx = 0
    explanation = explainer.explain_instance(X_train.iloc[sample_idx].values, model.predict_proba)
    feature_names = [f[0] for f in explanation.as_list()]
    feature_importance_values = [f[1] for f in explanation.as_list()]
    
    # SHAP 
    shap_explainer = shap.TreeExplainer(model)
    shap_values = shap_explainer.shap_values(X_train.iloc[sample_idx])
    shap_values_array = np.array(shap_values)
    shap_abs_mean = pd.DataFrame(np.mean(np.abs(shap_values_array), axis=1).flatten(), columns=['Importance'], index=X_train.columns)
    shap_abs_mean_sorted = shap_abs_mean.sort_values(by='Importance', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    
    with st.expander("Feature Importance using LIME and SHAP"):
        
        
        
        # LIME Feature Importances plot
    
        
        sns.barplot(x=feature_importance_values, y=feature_names, palette='rocket', ax=axes[0])
        axes[0].set_xlabel('Importance')
        axes[0].set_ylabel('Feature')
        axes[0].set_title('LIME Feature Importances')
        axes[0].invert_yaxis()

        # SHAP Feature Importances plot
        sns.barplot(data=shap_abs_mean_sorted, x='Importance', y=shap_abs_mean_sorted.index, palette='magma', ax=axes[1])
        axes[1].set_xlabel('Importance')
        axes[1].set_ylabel('Feature')
        axes[1].set_title('SHAP Feature Importances')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
    #### Understanding Model Explanations
    
    **LIME (Local Interpretable Model-agnostic Explanations):**
    - LIME helps explain individual predictions of machine learning models.
    - It trains simpler models on variations of the data to provide insights into why a model made a specific prediction for a single instance.
    - [Learn more about LIME](https://github.com/marcotcr/lime)
    
    **SHAP (SHapley Additive exPlanations):**
    - SHAP assigns each feature an importance value for the model's predictions across all instances.
    - It uses game theory to determine how much each feature contributes to the prediction.
    - [Learn more about SHAP](https://github.com/slundberg/shap)
    
    #### Interpreting SHAP Feature Importances
    - The SHAP plot visualizes the average impact of each feature on the model's output across all instances.
    - Positive SHAP values indicate features that contribute to higher predictions, while negative values indicate features that contribute to lower predictions.
    - Unlike LIME, SHAP does not show negative importance directly because it focuses on the additive contributions to the prediction rather than directionality.
        """)



def app():
    
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>⚕️National Poll on Healthy Aging</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h3>Training a Model to Predict of Number of Yearly Doctor Visits</h3>", unsafe_allow_html=True)


    # Load data
    global data
    data = load_data()
    

    st.write('##### We have used a "Random Forest Classifier" Model for prediction.')
    
    st.markdown("""
        Random Forest is a versatile machine learning algorithm that can be used for both classification and regression tasks. 
        It works by constructing multiple decision trees during training and outputs the class that is the most frequent in all the decision trees (classification) or takes the mean prediction (regression) of the individual trees.

        Random Forest is known for its robustness and ability to handle complex datasets with high-dimensional features. 
        It performs better compared to individual decision trees by averaging multiple trees.

        For more information about Random Forest: [Scikit-learn Random Forest Documentation](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)
            """)
    
    st.write("##### The following dropdowns gives you an overview of the Model's performance")


    # Train the original model
    model_original, X_train_original, X_test_original, y_train_original, y_test_original, y_pred_original = train_model(data, [])
    show_classification_report(model_original, y_test_original, y_pred_original)
    explain_lime_and_shap(X_train_original, model_original, y_train_original)
    
    

    st.write('##### Retrain the Model if you want a model without sensitive information:')
    
    st.markdown("""
         We aim to establish non-discrimination for protected or sensitive attribute(s) in our Model's behaviour. Also, it is often difficult to receive data with protected or sensitive attributes (like: gender, race, age etc.) due to a higher rate of refusal in the respondents. For such reasons, we also present you an option to create a model which does not use any sensitive attributes during its training process.
        
        """)
    
    # Layout for Drop header, checkboxes, and Retrain button in the same line
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.write("###### Drop Columns (Attributes):")
    with col2:
        drop_gender = st.checkbox('Gender', False)
    with col3:
        drop_race = st.checkbox('Race', False)

    # Prepare data for model training based on selected checkboxes
    features_to_drop = []
    if drop_gender:
        features_to_drop.append('Gender')
    if drop_race:
        features_to_drop.append('Race')

    # Join the features to drop into a single string for display
    features_str = "', '".join(features_to_drop)

    if st.button('Retrain Model'):
        model_retrained, X_train_retrained, X_test_retrained, y_train_retrained, y_test_retrained, y_pred_retrained = train_model(data, features_to_drop)

        st.write('### Here is how your new model performs:')
        
        # Print the values in the list
        st.write(f" Without using the attribute(s): '{features_str}' ")

        show_classification_report(model_retrained, y_test_retrained, y_pred_retrained)
        
        explain_lime_and_shap(X_train_retrained, model_retrained, y_train_retrained)
        
        
        
if __name__ == '__main__':
    app()
