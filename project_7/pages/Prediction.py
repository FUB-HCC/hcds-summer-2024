import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load('SVM_linear_scaled_p.sav')
scaler = joblib.load('scaler.pkl')
data = pd.read_csv('asthma_disease_data.csv')

features = data.drop(columns=['Diagnosis', 'PatientID', 'DoctorInCharge'])
target = data['Diagnosis']
scaled_features = scaler.transform(features)

main_header_style = "font-family:sans-serif; color:RoyalBlue; font-size: 26px; padding-bottom: 0;"
sub_header_style = "font-family:sans-serif; color:RoyalBlue; font-size: 22px; padding-bottom: 0;"

def init_session_state():
    if 'age' not in st.session_state:
        st.session_state.age = 25
    if 'gender' not in st.session_state:
        st.session_state.gender = 0
    if 'ethnicity' not in st.session_state:
        st.session_state.ethnicity = 0
    if 'education_level' not in st.session_state:
        st.session_state.education_level = 0
    if 'bmi' not in st.session_state:
        st.session_state.bmi = 22.5
    if 'smoking' not in st.session_state:
        st.session_state.smoking = 0
    if 'physical_activity' not in st.session_state:
        st.session_state.physical_activity = 5
    if 'diet_quality' not in st.session_state:
        st.session_state.diet_quality = 5
    if 'sleep_quality' not in st.session_state:
        st.session_state.sleep_quality = 7
    if 'pollution_exposure' not in st.session_state:
        st.session_state.pollution_exposure = 5
    if 'pollen_exposure' not in st.session_state:
        st.session_state.pollen_exposure = 5
    if 'dust_exposure' not in st.session_state:
        st.session_state.dust_exposure = 5
    if 'pet_allergy' not in st.session_state:
        st.session_state.pet_allergy = 0
    if 'family_history_asthma' not in st.session_state:
        st.session_state.family_history_asthma = 0
    if 'history_of_allergies' not in st.session_state:
        st.session_state.history_of_allergies = 0
    if 'eczema' not in st.session_state:
        st.session_state.eczema = 0
    if 'hay_fever' not in st.session_state:
        st.session_state.hay_fever = 0
    if 'gastro_reflux' not in st.session_state:
        st.session_state.gastro_reflux = 0
    if 'lung_function_fev1' not in st.session_state:
        st.session_state.lung_function_fev1 = 2.5
    if 'lung_function_fvc' not in st.session_state:
        st.session_state.lung_function_fvc = 3.5
    if 'wheezing' not in st.session_state:
        st.session_state.wheezing = 0
    if 'shortness_of_breath' not in st.session_state:
        st.session_state.shortness_of_breath = 0
    if 'chest_tightness' not in st.session_state:
        st.session_state.chest_tightness = 0
    if 'coughing' not in st.session_state:
        st.session_state.coughing = 0
    if 'nighttime_symptoms' not in st.session_state:
        st.session_state.nighttime_symptoms = 0
    if 'exercise_induced' not in st.session_state:
        st.session_state.exercise_induced = 0

init_session_state()

title = f'<h1 style="{main_header_style}">Prediction</h1>'
st.markdown(title, unsafe_allow_html=True)
st.write("---")

instruction = f"""
Please fill all the fields and then press the "Predict" button to see the result.
"""
st.markdown(instruction, unsafe_allow_html=True)

st.markdown(f'<h2 style="{sub_header_style}">Patient Information</h2>', unsafe_allow_html=True)
col1, col2, col3 = st.columns([3, 3, 3])
with col1:
    st.session_state.age = st.number_input("Age", 5, 80, st.session_state.age, help="Enter the age of the patient (5 to 80 years).")
with col2:
    st.session_state.gender = st.selectbox("Gender", [0, 1], index=st.session_state.gender, format_func=lambda x: "Male" if x == 0 else "Female", help="Select the gender of the patient.")
with col3:
    st.session_state.ethnicity = st.selectbox("Ethnicity", [0, 1, 2, 3], index=st.session_state.ethnicity, format_func=lambda x: ["Caucasian", "African American", "Asian", "Other"][x], help="Select the ethnicity of the patient.")

col4, col5 = st.columns([3, 3])
with col4:
    st.session_state.education_level = st.selectbox("Education Level", [0, 1, 2, 3], index=st.session_state.education_level, format_func=lambda x: ["None", "High School", "Bachelor's", "Higher"][x], help="Select the education level of the patient.")

st.markdown(f'<h2 style="{sub_header_style}">Lifestyle Factors</h2>', unsafe_allow_html=True)
col6, col7, col8 = st.columns([3, 3, 3])
with col6:
    st.session_state.bmi = st.number_input("BMI", 15.0, 40.0, st.session_state.bmi, help="Enter the Body Mass Index (BMI) of the patient (15 to 40).")
with col7:
    st.session_state.smoking = st.selectbox("Smoking Status", [0, 1], index=st.session_state.smoking, format_func=lambda x: "No" if x == 0 else "Yes", help="Select the smoking status of the patient. If the patient smokes every day or smoked every day in the past, choose Yes. [For more info](https://www.cdc.gov/nchs/nhis/tobacco/tobacco_glossary.htm#:~:text=Amount%20smoked%3A%20The%20average%20number,and%20who%20currently%20smokes%20cigarettes.)")
with col8:
    st.session_state.physical_activity = st.number_input("Weekly Physical Activity (hours)", 0, 10, st.session_state.physical_activity, help="Enter the weekly physical activity of the patient in hours (0 to 10).")

col9, col10 = st.columns([3, 3])
with col9:
    st.session_state.diet_quality = st.number_input("Diet Quality", 0, 10, st.session_state.diet_quality, help="Enter the diet quality score of the patient (0 to 10). 10 is according to the current optimal daily intake to cover all nutrition needs and 1 is a very poor diet. [For more info](https://www.iaea.org/topics/diet-quality#:~:text=Diet%20quality%20refers%20to%20a,a%20healthy%20and%20active%20life.)")
with col10:
    st.session_state.sleep_quality = st.number_input("Sleep Quality", 4, 10, st.session_state.sleep_quality, help="Enter the sleep quality score of the patient (4 to 10). Sleep quality refers to factors like sleep efficiency and latency, so 10 would be without any disturbances and 4 with many disturbances during sleep. [For more info](https://www.thensf.org/what-is-sleep-quality/)")

st.markdown(f'<h2 style="{sub_header_style}">Environmental and Allergy Factors</h2>', unsafe_allow_html=True)
col11, col12, col13 = st.columns([3, 3, 3])
with col11:
    st.session_state.pollution_exposure = st.number_input("Pollution Exposure", 0, 10, st.session_state.pollution_exposure, help="Enter the pollution exposure score of the patient (0 to 10). 0 would be a life environment with minimal exposure and very low industrial activity, 10 would be extreme exposure with major industrial zones and cities with poor air quality standards.")
with col12:
    st.session_state.pollen_exposure = st.number_input("Pollen Exposure", 0, 10, st.session_state.pollen_exposure, help="Enter the pollen exposure score of the patient (0 to 10). 0 would be low pollen counts where there is no significant presence of pollen-producing plants or trees, 10 would be extremely high pollen counts with constant presence of numerous pollen-producing plants and trees throughout the year.")
with col13:
    st.session_state.dust_exposure = st.number_input("Dust Exposure", 0, 10, st.session_state.dust_exposure, help="Enter the dust exposure score of the patient (0 to 10). Dust sources can be traffic roads, construction sites, or industrial activities, as well as deserts.  0 would be minimal dust exposure, 10 would be extreme dust exposure in highly industrial or desert areas.")

col14, col15 = st.columns([3, 3])
with col14:
    st.session_state.pet_allergy = st.selectbox("Pet Allergy", [0, 1], index=st.session_state.pet_allergy, format_func=lambda x: "No" if x == 0 else "Yes", help="Select whether the patient has a pet allergy. All pets included.")

st.markdown(f'<h2 style="{sub_header_style}">Medical History</h2>', unsafe_allow_html=True)
col16, col17, col18 = st.columns([3, 3, 3])
with col16:
    st.session_state.family_history_asthma = st.selectbox("Family History of Asthma", [0, 1], index=st.session_state.family_history_asthma, format_func=lambda x: "No" if x == 0 else "Yes", help="Select whether the patient has a family history of asthma. This includes parental, grandparental, and sibling history. [For more info](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1618803/)")
with col17:
    st.session_state.history_of_allergies = st.selectbox("History of Allergies", [0, 1], index=st.session_state.history_of_allergies, format_func=lambda x: "No" if x == 0 else "Yes", help="Select whether the patient has a history of allergies.")
with col18:
    st.session_state.eczema = st.selectbox("Eczema", [0, 1], index=st.session_state.eczema, format_func=lambda x: "No" if x == 0 else "Yes", help="Select whether the patient has eczema.")

col19, col20 = st.columns([3, 3])
with col19:
    st.session_state.hay_fever = st.selectbox("Hay Fever", [0, 1], index=st.session_state.hay_fever, format_func=lambda x: "No" if x == 0 else "Yes", help="Select whether the patient has hay fever.")
with col20:
    st.session_state.gastro_reflux = st.selectbox("Gastroesophageal Reflux", [0, 1], index=st.session_state.gastro_reflux, format_func=lambda x: "No" if x == 0 else "Yes", help="Select whether the patient has gastroesophageal reflux.")

st.markdown(f'<h2 style="{sub_header_style}">Clinical Measurements</h2>', unsafe_allow_html=True)
col21, col22 = st.columns([3, 3])
with col21:
    st.session_state.lung_function_fev1 = st.number_input("Lung Function (FEV1)", 1.0, 4.0, st.session_state.lung_function_fev1, help="Enter the forced expiratory volume in 1 second (FEV1) of the patient (1.0 to 4.0 liters).")
with col22:
    st.session_state.lung_function_fvc = st.number_input("Lung Function (FVC)", 1.5, 6.0, st.session_state.lung_function_fvc, help="Enter the forced vital capacity (FVC) of the patient (1.5 to 6.0 liters).")

st.markdown(f'<h2 style="{sub_header_style}">Symptoms</h2>', unsafe_allow_html=True)
col23, col24, col25 = st.columns([3, 3, 3])
with col23:
    st.session_state.wheezing = st.selectbox("Wheezing", [0, 1], index=st.session_state.wheezing, format_func=lambda x: "No" if x == 0 else "Yes", help="Select whether the patient has wheezing.")
with col24:
    st.session_state.shortness_of_breath = st.selectbox("Shortness of Breath", [0, 1], index=st.session_state.shortness_of_breath, format_func=lambda x: "No" if x == 0 else "Yes", help="Select whether the patient has shortness of breath.")
with col25:
    st.session_state.chest_tightness = st.selectbox("Chest Tightness", [0, 1], index=st.session_state.chest_tightness, format_func=lambda x: "No" if x == 0 else "Yes", help="Select whether the patient has chest tightness.")

col26, col27, col28 = st.columns([3, 3, 3])
with col26:
    st.session_state.coughing = st.selectbox("Coughing", [0, 1], index=st.session_state.coughing, format_func=lambda x: "No" if x == 0 else "Yes", help="Select whether the patient has coughing.")
with col27:
    st.session_state.nighttime_symptoms = st.selectbox("Nighttime Symptoms", [0, 1], index=st.session_state.nighttime_symptoms, format_func=lambda x: "No" if x == 0 else "Yes", help="Select whether the patient has nighttime symptoms.")
with col28:
    st.session_state.exercise_induced = st.selectbox("Exercise Induced Symptoms", [0, 1], index=st.session_state.exercise_induced, format_func=lambda x: "No" if x == 0 else "Yes", help="Select whether the patient has exercise-induced symptoms.")

input_data = np.array([
    st.session_state.age, st.session_state.gender, st.session_state.ethnicity, st.session_state.education_level, st.session_state.bmi, st.session_state.smoking, st.session_state.physical_activity, 
    st.session_state.diet_quality, st.session_state.sleep_quality, st.session_state.pollution_exposure, st.session_state.pollen_exposure, st.session_state.dust_exposure, 
    st.session_state.pet_allergy, st.session_state.family_history_asthma, st.session_state.history_of_allergies, st.session_state.eczema, st.session_state.hay_fever, 
    st.session_state.gastro_reflux, st.session_state.lung_function_fev1, st.session_state.lung_function_fvc, st.session_state.wheezing,
    st.session_state.shortness_of_breath, st.session_state.chest_tightness, st.session_state.coughing, st.session_state.nighttime_symptoms, st.session_state.exercise_induced
]).reshape(1, -1)

scaled_input_data = scaler.transform(input_data)

button_css = """
<style>
div.stButton > button:first-child {
    background-color: green;
    color: white;
    font-size: 20px;
    border-radius: 10px;
    padding: 10px 30px;
    display: block;
    margin: 0 auto;
}
</style>
"""
st.markdown(button_css, unsafe_allow_html=True)

if st.button("Predict"):
    prediction = model.predict(scaled_input_data)[0]
    prediction_proba = model.predict_proba(scaled_input_data)[0]

    result = "Positive" if prediction == 1 else "Negative"
    probability = max(prediction_proba) * 100

    if result == "Positive":
        st.markdown(f"""
            <div style='background-color: #ffcccc; padding: 20px; border-radius: 10px; text-align: center;'>
                <h3 style='color: black;'>Result: <span style='color: red;'>{result}</span> with {probability:.2f}% probability</h3>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style='background-color: #ccffcc; padding: 20px; border-radius: 10px; text-align: center;'>
                <h3 style='color: black;'>Result: <span style='color: green;'>{result}</span> with {probability:.2f}% probability</h3>
            </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""If the probability of the prediction is lower than 80%, please be cautious with the prediction and do not rely on it.""", unsafe_allow_html=True)
    st.markdown(f"""
    Please also consider the graph for similarity check with the training dataset and the explanation of the prediction below.
    """, unsafe_allow_html=True)

    st.markdown(f'<h2 style="{main_header_style}">Dataset Similarity Check</h2>', unsafe_allow_html=True)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)

    new_point_pca = pca.transform(scaled_input_data)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=target, cmap='viridis', alpha=0.6, edgecolors='w', s=50)
    plt.scatter(new_point_pca[:, 0], new_point_pca[:, 1], c='red', marker='X', s=100, edgecolors='black')

    plt.title('Dataset Similarity Check')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    handles, labels = scatter.legend_elements(prop="colors")
    legend_labels = ["Negative Diagnosis", "Positive Diagnosis"]
    handles = [handles[0], handles[1], plt.Line2D([0], [0], marker='X', color='w', label='New Input Point', markerfacecolor='red', markersize=15, markeredgewidth=2, markeredgecolor='black')]
    plt.legend(handles, legend_labels + ["New Input Point"], loc='upper right')

    st.pyplot(plt)

    pca_interpretation = f"""
    The scatter plot above shows the distribution of the existing data points in the dataset (colored by their diagnosis status) and the new input point (marked with a red 'X').
    <ul>
        <li><strong>Similarity Check:</strong> If the new input point is close to other points, it indicates that the new input is similar to those points in the dataset.</li>
        <li><strong>Interpretation:</strong> Points clustered together share similar characteristics. The new input point's position can give an idea of whether the patient's data is similar to others with asthma (if clustered with positive cases) or not (if clustered with negative cases). In general, you can trust the prediction more, when your input lays within the dense area of data points, and less when it is isolated from the other points.</li>
    </ul>
    """
    st.markdown(pca_interpretation, unsafe_allow_html=True)

    st.markdown(f'<h2 style="{main_header_style}">Explanation of Prediction</h2>', unsafe_allow_html=True)
    contributions = scaled_input_data.flatten() * model.coef_[0]
    contributions_df = pd.DataFrame({
        'Feature': features.columns,
        'Contribution': contributions
    }).sort_values(by='Contribution', ascending=False)

    st.markdown(f"""
    The plot below shows the contribution of each feature to the prediction above (local explanation). This is calculated as the product of the scaled feature value and its coefficient.
    """, unsafe_allow_html=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Contribution', y='Feature', data=contributions_df, palette='viridis')
    plt.title('Explanation of Prediction')
    plt.xlabel('Contribution')
    plt.ylabel('Feature')

    for i, (value, name) in enumerate(zip(contributions_df['Contribution'], contributions_df['Feature'])):
        plt.text(value, i, f'{value:.2f}', va='center', ha='right' if value < 0 else 'left', color='black')

    plt.gca().xaxis.set_visible(False)
    plt.grid(False)

    st.pyplot(plt)

    explanation = f"""
    Features with higher contributions are more influential in the prediction. This helps to understand which features are pushing the prediction towards a positive or negative diagnosis.
    """
    st.markdown(explanation, unsafe_allow_html=True)
