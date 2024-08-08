import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import seaborn as sns
import numpy as np
import base64
import io
from sklearn.model_selection import train_test_split



# Define mappings for categorical features
mappings = {
    'Number of Doctors Visited': {
        1: '0-1 doctors', 2: '2-3 doctors', 3: '4 or more doctors'
    },
    'Physical Health': {
        -1: 'Refused', 1: 'Excellent', 2: 'Very Good', 3: 'Good', 4: 'Fair', 5: 'Poor'
    },
    'Mental Health': {
        -1: 'Refused', 1: 'Excellent', 2: 'Very Good', 3: 'Good', 4: 'Fair', 5: 'Poor'
    },
    'Dental Health': {
        -1: 'Refused', 1: 'Excellent', 2: 'Very Good', 3: 'Good', 4: 'Fair', 5: 'Poor', 6: 'N/A - I have dentures'
    },
    'Employment': {
        -1: 'Refused', 1: 'Working full-time', 2: 'Working part-time', 3: 'Retired', 4: 'Not working at this time'
    },
    'Stress Keeps Patient from Sleeping': {
        0: 'No', 1: 'Yes'
    },
    'Medication Keeps Patient from Sleeping': {
        0: 'No', 1: 'Yes'
    },
    'Pain Keeps Patient from Sleeping': {
        0: 'No', 1: 'Yes'
    },
    'Bathroom Needs Keeps Patient from Sleeping': {
        0: 'No', 1: 'Yes'
    },
    'Unknown Keeps Patient from Sleeping': {
        0: 'No', 1: 'Yes'
    },
    'Trouble Sleeping': {
        -1: 'Refused', 1: '0 nights', 2: '1-2 nights', 3: '3-5 nights'
    },
    'Prescription Sleep Medication': {
        -1: 'Refused', 1: 'Use regularly', 2: 'Use occasionally', 3: 'Do not use'
    },
    'Race': {
        -2: 'Not asked', -1: 'Refused', 1: 'White, Non-Hispanic', 2: 'Black, Non-Hispanic', 3: 'Other, Non-Hispanic',
        4: 'Hispanic', 5: '2+ Races, Non-Hispanic'
    },
    'Gender': {
        -2: 'Not asked', -1: 'Refused', 1: 'Male', 2: 'Female'
    },
        'Age': {1: '50-64 years', 2: '65-80 years'}
}

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('NPHA-doctor-visits.csv')
    return data

def fig_to_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    fig_str = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{fig_str}" />'


# Function to train a model
def train_model(data, features_to_drop):
    X = data.drop(['Number of Doctors Visited'] + features_to_drop, axis=1)
    y = data['Number of Doctors Visited']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


    
def predict_with_original_model():
    input_data = {}
    st.write("###### (uses all features)")
    st.write("#### Choose Values from Dropdown")

    # Demographic Attributes
    st.write("##### Demographic Attributes")
    demographic_attributes = ['Age', 'Race', 'Gender', 'Employment']
    cols_per_row_demo = 4
    for i, feature in enumerate(demographic_attributes):
        if i % cols_per_row_demo == 0:
            cols = st.columns(cols_per_row_demo)
        unique_values = sorted(data[feature].unique())
        mapped_values = [mappings[feature][value] for value in unique_values]
        with cols[i % cols_per_row_demo]:
            input_data[feature] = st.selectbox(f'{feature}', mapped_values)

    # Health Status
    st.write("##### Health Status")
    health_status = ['Physical Health', 'Mental Health', 'Dental Health']
    cols_per_row_health = 3
    for i, feature in enumerate(health_status):
        if i % cols_per_row_health == 0:
            cols = st.columns(cols_per_row_health)
        unique_values = sorted(data[feature].unique())
        mapped_values = [mappings[feature][value] for value in unique_values]
        with cols[i % cols_per_row_health]:
            input_data[feature] = st.selectbox(f'{feature}', mapped_values)

    # Sleep Status
    st.write("##### Sleep Status")
    sleep_status = [
        'Stress Keeps Patient from Sleeping', 'Medication Keeps Patient from Sleeping',
        'Pain Keeps Patient from Sleeping', 'Bathroom Needs Keeps Patient from Sleeping',
        'Unknown Keeps Patient from Sleeping', 'Trouble Sleeping', 'Prescription Sleep Medication'
    ]
    cols_per_row_sleep = 4
    for i, feature in enumerate(sleep_status):
        if i % cols_per_row_sleep == 0:
            cols = st.columns(cols_per_row_sleep)
        unique_values = sorted(data[feature].unique())
        mapped_values = [mappings[feature][value] for value in unique_values]
        with cols[i % cols_per_row_sleep]:
            input_data[feature] = st.selectbox(f'{feature}', mapped_values)

    # Convert input_data back to original numeric values
    for feature in input_data:
        reverse_mapping = {v: k for k, v in mappings[feature].items()}
        input_data[feature] = reverse_mapping[input_data[feature]]

    # Predict based on selected model
    if st.button('Predict (with Original Model)', key='predict_button'):
        input_df = pd.DataFrame([input_data])

        input_df = input_df[X_test_original.columns]

        prediction = model_original.predict(input_df)
        mapped_prediction = mappings['Number of Doctors Visited'].get(prediction[0], 'Unknown')

        st.markdown(f'### This person is predicted to visit <span style="text-decoration: underline;">{mapped_prediction}</span> in a year (based on Retrained Model with removed feature(s)). Below you can find the probability of the person belonging to any of the three classes.', unsafe_allow_html=True)

        
        st.markdown(
            f'#### Under "Current Prediction", you will see the prediction of this model based on your chosen values at the moment. To help you better understand how changing one or more attribute values affect the prediction, we encourage you to change value(s) from the dropdowns and make another prediction. If you choose to make another prediction with changed value(s), a second section named "Previous Prediction" will appear (along with "Current Prediction"), which shows you your last prediction results, and helps you compare how the probability of each predicted class has changed. ')


        probabilities = model_original.predict_proba(input_df)[0]

        # Display the probabilities in a pie chart
        labels = [mappings['Number of Doctors Visited'][cls] for cls in model_original.classes_]
        colors = cm.get_cmap('Set3')(np.linspace(0, 1, len(labels)))
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(probabilities, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 8})
        plt.title('Probability Distribution of Classes', fontsize=8, fontweight='bold')
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Convert figure to HTML and store in session state
        fig_html = fig_to_html(fig)
        
        # Store the current prediction in session state
        if 'current_prediction' in st.session_state:
            st.session_state['previous_prediction'] = st.session_state['current_prediction']

        st.session_state['current_prediction'] = fig_html
                    
        st.markdown("<br><br>", unsafe_allow_html=True)  # Add blank space
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('### Current Prediction')
            st.write(st.session_state['current_prediction'], unsafe_allow_html=True)
        with col2:
            if 'previous_prediction' in st.session_state:
                st.markdown('### Previous Prediction')
                st.write(st.session_state['previous_prediction'], unsafe_allow_html=True)



def predict_with_retrained_model():
    input_data_retrained = {}
    
    # Section for selecting values from dropdowns
    st.write("###### (does not use removed feature(s))")
    st.write("### Choose Values from Dropdown")
    
    # Demographic Attributes
    st.write("#### Demographic Attributes")
    demographic_attributes = ['Age', 'Race', 'Gender', 'Employment']
    cols_per_row_demo = 4
    for i, feature in enumerate(demographic_attributes):
        if i % cols_per_row_demo == 0:
            cols = st.columns(cols_per_row_demo)
        unique_values = sorted(data[feature].unique())
        mapped_values = [mappings[feature][value] for value in unique_values]
        with cols[i % cols_per_row_demo]:
            input_data_retrained[feature] = st.selectbox(f'{feature}', mapped_values)

    # Health Status
    st.write("#### Health Status")
    health_status = ['Physical Health', 'Mental Health', 'Dental Health']
    cols_per_row_health = 3
    for i, feature in enumerate(health_status):
        if i % cols_per_row_health == 0:
            cols = st.columns(cols_per_row_health)
        unique_values = sorted(data[feature].unique())
        mapped_values = [mappings[feature][value] for value in unique_values]
        with cols[i % cols_per_row_health]:
            input_data_retrained[feature] = st.selectbox(f'{feature}', mapped_values)

    # Sleep Status
    st.write("#### Sleep Status")
    sleep_status = [
        'Stress Keeps Patient from Sleeping', 'Medication Keeps Patient from Sleeping',
        'Pain Keeps Patient from Sleeping', 'Bathroom Needs Keeps Patient from Sleeping',
        'Unknown Keeps Patient from Sleeping', 'Trouble Sleeping', 'Prescription Sleep Medication'
    ]
    cols_per_row_sleep = 4
    for i, feature in enumerate(sleep_status):
        if i % cols_per_row_sleep == 0:
            cols = st.columns(cols_per_row_sleep)
        unique_values = sorted(data[feature].unique())
        mapped_values = [mappings[feature][value] for value in unique_values]
        with cols[i % cols_per_row_sleep]:
            input_data_retrained[feature] = st.selectbox(f'{feature}', mapped_values)

    # Convert input_data back to original numeric values
    for feature in input_data_retrained:
        reverse_mapping = {v: k for k, v in mappings[feature].items()}
        input_data_retrained[feature] = reverse_mapping[input_data_retrained[feature]]

    # Train and predict based on retrained model
    if st.button('Predict (with Retrained Model)', key='train_predict_button_retrained'):
        global model_retrained
        
        # Dropping selected features directly here
        features_to_drop = []
        if 'Gender' not in input_data_retrained:
            features_to_drop.append('Gender')
        if 'Race' not in input_data_retrained:
            features_to_drop.append('Race')
        
        
        # Predict based on retrained model
        input_df_retrained = pd.DataFrame([input_data_retrained])
        input_df_retrained = input_df_retrained[X_test_retrained.columns]
        prediction_retrained = model_retrained.predict(input_df_retrained)
        mapped_prediction = mappings['Number of Doctors Visited'].get(prediction_retrained[0], 'Unknown')
        st.markdown(f'### This person is predicted to visit <span style="text-decoration: underline;">{mapped_prediction}</span> in a year (based on Retrained Model with removed feature(s)). Below you can find the probability of the person belonging to any of the three classes.', unsafe_allow_html=True)

        
        st.markdown(
            f'#### Under "Current Prediction", you will see the prediction of this model based on your chosen values at the moment. To help you better understand how changing one or more attribute values affect the prediction, we encourage you to change value(s) from the dropdowns and make another prediction. If you choose to make another prediction with changed value(s), a second section named "Previous Prediction" will appear (along with "Current Prediction"), which shows you your last prediction results, and helps you compare how the probability of each predicted class has changed.')

        probabilities_retrained = model_retrained.predict_proba(input_df_retrained)[0]

        # Display the probabilities in a pie chart for current prediction
        labels = [mappings['Number of Doctors Visited'][cls] for cls in model_retrained.classes_]
        colors = cm.get_cmap('Set3')(np.linspace(0, 1, len(labels)))
        fig_current, ax_current = plt.subplots(figsize=(4, 4))
        ax_current.pie(probabilities_retrained, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 8})
        plt.title('Probability Distribution of Classes (Current Prediction)', fontsize=8, fontweight='bold')
        ax_current.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        
        # Convert figure to HTML and store in session state for current prediction
        fig_current_html = fig_to_html(fig_current)
        if 'current_prediction_retrained' in st.session_state:
            st.session_state['previous_prediction_retrained'] = st.session_state['current_prediction_retrained']
        st.session_state['current_prediction_retrained'] = fig_current_html
        
        # Display the current prediction
        st.markdown("<br><br>", unsafe_allow_html=True)  # Add blank space
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('### Current Prediction (Retrained Model)')
            st.write(st.session_state['current_prediction_retrained'], unsafe_allow_html=True)
        
        # Display the previous prediction if exists
        with col2:
            if 'previous_prediction_retrained' in st.session_state:
                st.markdown('### Previous Prediction (Retrained Model)')
                st.write(st.session_state['previous_prediction_retrained'], unsafe_allow_html=True)


                
def app():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>⚕️National Poll on Healthy Aging</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>Prediction of Number of Yearly Doctor Visits</h2>", unsafe_allow_html=True)

    global data
    data = load_data()
    
    # Initialize session state variables if they don't exist
    if 'retrained' not in st.session_state:
        st.session_state.retrained = False
    if 'retrain_second_flag' not in st.session_state:
        st.session_state.retrain_second_flag = False
        
    # Initialize session state variables
    if 'current_prediction_html' not in st.session_state:
        st.session_state.current_prediction_html = None
    if 'previous_prediction_html' not in st.session_state:
        st.session_state.previous_prediction_html = None


    # Train the original model
    global model_original, X_original, y_original, X_train_original, X_test_original, y_train_original, y_test_original, model_retrained, X_train_retrained, X_test_retrained,y_train_retrained, y_test_retrained 
    
    
    model_original, X_train_original, X_test_original, y_train_original, y_test_original = train_model(data, [])

    
    # Layout for Drop header, checkboxes, and Retrain button in the same line
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.write("##### Drop Columns with Sensitive Information:")
    with col2:
        drop_gender = st.checkbox('Gender', False)
    with col3:
        drop_race = st.checkbox('Race', False)

    # Prepare data for model training based on selected checkboxes
    global features_to_drop
    
    features_to_drop = []
    if drop_gender:
        features_to_drop.append('Gender')
    if drop_race:
        features_to_drop.append('Race')

    
    
    
    if st.button('Retrain Model'):       
        
        model_retrained, X_train_retrained, X_test_retrained,y_train_retrained, y_test_retrained = train_model(data, features_to_drop)
        st.write('### <span style="color: #007BFF;">Model retrained successfully!</span>', unsafe_allow_html=True)
        st.session_state.retrain_second_flag = True
        
        
    st.markdown("<br>", unsafe_allow_html=True)  
    # Option to predict using either original or retrained model
    st.write('#### Make Your Own Prediction')
#     prediction_model = st.selectbox('Predict output from:', ['Original Model', 'Retrained Model'])

    st.markdown("<br>", unsafe_allow_html=True)  # Add blank space
    #st.markdown('### <span style="color: #007BFF;">Select Values to Make Your Own Prediction</span>', unsafe_allow_html=True)


    # Dropdown to select model for prediction
    prediction_model = st.selectbox('Predict output from:', ['Original Model', 'Retrained Model'])
    
    if prediction_model == 'Retrained Model' and st.session_state.retrain_second_flag:
        st.session_state.retrained = True

    if prediction_model == 'Original Model' and not st.session_state.retrain_second_flag:
        predict_with_original_model()

    elif prediction_model == 'Original Model' and st.session_state.retrain_second_flag:
        predict_with_original_model()

    elif prediction_model == 'Retrained Model' and st.session_state.retrain_second_flag:
        predict_with_retrained_model()

    elif prediction_model == 'Retrained Model' and not st.session_state.retrain_second_flag:
        st.markdown('### Sorry, Please retrain the model to use the "Retrained Model" option for prediction.')
        
        
    

if __name__ == '__main__':
    app()
