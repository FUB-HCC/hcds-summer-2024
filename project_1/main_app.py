import streamlit as st

# Import your pages here
import welcome
import npha
import barplot
import model
import predict
import bias

# Dictionary to map page names to functions
PAGES = {
    "Welcome": welcome,
    "About the Dataset": npha,
    "Exploring Data": barplot,
    "Training A Model": model,
    "Making A Prediction" : predict,
    "Bias Correction" : bias
}

st.sidebar.title('Go To')
selection = st.sidebar.radio("Sections", list(PAGES.keys()))

page = PAGES[selection]
page.app()
