import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

models = ["RandomForest","LogisticRegression", "DecisionTree", "NeuralNetwork"]
ports = {
    "RandomForest": 8054,
    "LogisticRegression": 8051,
    "DecisionTree": 8052,
    "NeuralNetwork": 8053
}

st.set_page_config(layout="wide")
st.sidebar.title("Model Selection")

model_name = st.sidebar.selectbox("Choose a model", models)

st.title(f"{model_name} Explainer Dashboard")


port = ports[model_name]


st.markdown(
    f"""
    <iframe src="http://localhost:{port}" width="100%" height="800px" frameborder="0"></iframe>
    """,
    unsafe_allow_html=True
)
fairness_result = pd.read_csv(f'{model_name}_fairness_result.csv')


categories = [
    "Dynamic Activity", "Static Activity",
    "Fitbit and Dynamic Activity", "Fitbit and Static Activity",
    "Apple Watch and Dynamic Activity", "Apple Watch and Static Activity"
]

female_probabilities = [
    fairness_result['probability_female_with_dynamic_activity'][0],
    fairness_result['probability_female_with_static_activity'][0],
    fairness_result['probability_female_with_device_fitbit_and_dynamic_activity'][0],
    fairness_result['probability_female_with_device_fitbit_and_static_activity'][0],
    fairness_result['probability_female_with_device_apple_watch_and_dynamic_activity'][0],
    fairness_result['probability_female_with_device_apple_watch_and_static_activity'][0]
]

male_probabilities = [
    fairness_result['probability_male_with_dynamic_activity'][0],
    fairness_result['probability_male_with_static_activity'][0],
    fairness_result['probability_male_with_device_fitbit_and_dynamic_activity'][0],
    fairness_result['probability_male_with_device_fitbit_and_static_activity'][0],
    fairness_result['probability_male_with_device_apple_watch_and_dynamic_activity'][0],
    fairness_result['probability_male_with_device_apple_watch_and_static_activity'][0]
]

predictive_parity = fairness_result['predictive_parity'][0]
error_rate = fairness_result['error_rate'][0]


fig, axs = plt.subplots(2, 3, figsize=(18, 12))


axs = axs.flatten()


for i, category in enumerate(categories):
    df = pd.DataFrame({
        'Gender': ['Female', 'Male'],
        'Probability': [female_probabilities[i], male_probabilities[i]]
    })
    sns.barplot(x='Gender', y='Probability', data=df, ax=axs[i])
    axs[i].set_title(category)
    axs[i].set_ylim(0, 1) 


plt.tight_layout()


st.pyplot(fig)


fig2, ax = plt.subplots(figsize=(10, 6))
ax.bar(['Predictive Parity', 'Error Rate'], [predictive_parity, error_rate], color=['red', 'green'])
ax.set_ylim(0, 1)  
ax.set_title('Predictive Parity and Error Rate')
ax.set_ylabel('Rate')


st.pyplot(fig2)
st.write("Use the explainer dashboard to understand the model's predictions.")
