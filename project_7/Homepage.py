import streamlit as st
from st_pages import Page, show_pages, add_page_title

show_pages(
    [
        Page("Homepage.py", "Home", "🏠"),
        Page("pages/Prediction.py", "Prediction", "▶️"),
        Page("pages/AboutDs.py", "About dataset", "📊"),
        Page("pages/AboutModel.py", "About model", "🤖"),
        Page("pages/Feedback.py", "Feedback", "💌")
    ]
)

title = '<h1 style="font-family:sans-serif; color:RoyalBlue; font-size: 26px; padding-bottom: 0;">Welcome to the Asthma Diagnosis Aid Tool for Doctors!</h1>'
st.markdown(title, unsafe_allow_html=True)
st.write("---")
st.image('img.jpg')
st.write("This application is designed to assist healthcare providers in diagnosing asthma using patient data.")

# Disclaimer
st.markdown('<h5>Disclaimer:</h5>', unsafe_allow_html=True)
disclaimer = """
<style>
    .disclaimer {
        background-color: #f9f9f9;
        border-left: 6px solid #f44336;
        padding: 10px;
    }
</style>
<div class="disclaimer">
    <p style="font-family:sans-serif; color:black; font-size: 16px;">
    This Asthma Prediction tool is an aid tool and should not be fully relied upon for diagnosis. It is intended to assist doctors by providing additional insights. Please consider the points below before starting to use the tool:
    </p>
    <ol style="font-family:sans-serif; color:black; font-size: 16px;">
        <li>When in doubt about the fairness of the results, alter the values of protected variables (e.g., gender, ethnicity) to see if the result changes significantly.</li>
        <li>Review the distribution of the training dataset used before using the tool for reliability check (e.g., ensure the ethnicity group is included or check if the patient age is too old/young compared to the model’s training dataset).</li>
        <li>Needed test results: Ensure that the patient has undergone the necessary tests (e.g., Lung Function FEV1 and FVC) as they are required for the model's prediction.</li>
        <li>For checking if the model aligns with medical knowledge, please refer to the global explanation section in the 'About Model' page to understand which variables were most significant for prediction.</li>
    </ol>
</div>
"""
st.markdown(disclaimer, unsafe_allow_html=True)
st.write("")

st.markdown('<h5>How to Use:</h5>', unsafe_allow_html=True)
st.write("1. Navigate to the Main Menu located in the top-left corner of the screen.")
st.write("2. Click on the 'Prediction' tab to access the prediction tool.")
st.write("3. Enter relevant information as requested in the input fields on the Prediction page in order to get a prediction.")
st.write("4. Click on the 'Predict' button to obtain predictions based on the provided data.")

st.write("")

st.markdown('<h5>Page Overview:</h5>', unsafe_allow_html=True)
st.markdown("- **Prediction**: Enter patient data to predict the likelihood of asthma.")
st.markdown("- **About Dataset**: Overview of the training dataset, including key features and statistics.")
st.markdown("- **About Model**: Information about the model's performance, fairness evaluation regarding protected variables, and other model card details.")
st.markdown("- **Feedback & Contact**: Provide feedback or contact us for support.")

st.write("")

