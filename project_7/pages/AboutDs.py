import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import squarify

df = pd.read_csv("asthma_recoded.csv", index_col=False)

variable_descriptions = {
    'PatientID': 'Unique identifier for each patient', #hidden
    'Age': 'Age of the patient in years',
    'Gender': 'Gender of the patient (Male or Female)',
    'Ethnicity': 'Ethnicity of the patient (Caucasian, African American, Asian, or Other).',
    'EducationLevel': 'Education level of the patient (No education, High School, Bachelors, or Higher)',
    'BMI': 'Body Mass Index of the patient. Beware, that this is BMI, not the weight of a patient',
    'Smoking': 'Smoking status of the patient (Non-smoker or Smoker). Beware, that underlying measurements of these categories were not provided by the authors of the dataset',
    'PhysicalActivity': 'Weekly physical activity in hours (from 0 to 10)',
    'DietQuality': 'Diet quality score (0 to 10)', #hidden
    'SleepQuality': 'Sleep quality score (4 to 10)', #hidden
    'PollutionExposure': 'Exposure to pollution (0 to 10)', #hidden
    'PollenExposure': 'Exposure to pollen (0 to 10)', #hidden
    'DustExposure': 'Exposure to dust (0 to 10)', #hidden
    'PetAllergy': 'Presence of pet allergy (Yes or No)',
    'FamilyHistoryAsthma': 'Presence of family history of asthma (Yes or No)',
    'HistoryOfAllergies': 'Presence of history of allergies (Yes or No)',
    'Eczema': 'Presence of eczema (Yes or No)',
    'HayFever': 'Presence of hay fever (Yes or No)',
    'GastroesophagealReflux': 'Presence of gastroesophageal reflux (Yes or No)',
    'LungFunctionFEV1': 'Forced Expiratory Volume in 1 second (from 1.0 to 4.0 liters)',
    'LungFunctionFVC': 'Forced Vital Capacity (from 1.5 to 6.0 liters)',
    'Wheezing': 'Presence of wheezing (Yes or No)',
    'ShortnessOfBreath': 'Presence of shortness of breath (Yes or No)',
    'ChestTightness': 'Presence of chest tightness (Yes or No)',
    'Coughing': 'Presence of coughing (Yes or No)',
    'NighttimeSymptoms': 'Presence of nighttime symptoms (Yes or No). Beware, that there was no definition of a nighttime symptom provided.',
    'ExerciseInduced': 'Presence of symptoms induced by exercise (Yes or No)',
    'Diagnosis': 'Diagnosis status for asthma (Yes if a patient has asthma and No if they are healthy). This is a target of the model, i.e. what the model predicts',
    'DoctorInCharge': 'Doctor in charge' #hidden
}

variables_to_hide = ['PatientID', 'DietQuality', 'SleepQuality', 'PollutionExposure', 'PollenExposure', 'DustExposure', 'DoctorInCharge']

title = '<h1 style="font-family:sans-serif; color:RoyalBlue; font-size: 26px; padding-bottom: 0;">About Dataset</h1>'
st.markdown(title, unsafe_allow_html=True)
st.write("---")
st.write("The model was trained on [*The Asthma Disease Dataset*](https://www.kaggle.com/datasets/rabieelkharoua/asthma-disease-dataset/data), which was firstly published by Rabie El Kharoua on Kaggle in 2024. The original dataset contains extensive health information for 2392 patients, including their demographic traits, lifestyle habits, medical history, relevant clinical measurements, symptoms and diagnosis.")
st.write("In the drop-down menu below, you can choose the feature of interest and obtain its description and distribution plot. In the plots, the number of patients with asthma per different categories is also provided.")

selected_variable = st.selectbox('Select a variable to explore:', [col for col in df.columns if col not in variables_to_hide])

if selected_variable:
    st.write(variable_descriptions[selected_variable])

    category_order = ['Asthma', 'Healthy']
    colors = {
        'Healthy': 'lightblue',
        'Asthma': 'lightpink'
    }

    fig = px.histogram(df, x=selected_variable, color="Diagnosis", category_orders={"Diagnosis": category_order},
                       color_discrete_map=colors)

    fig.update_layout(legend_title_text='Diagnosis', legend_tracegroupgap=10)

    st.plotly_chart(fig)



