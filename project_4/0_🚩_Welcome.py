
from streamlit_option_menu import option_menu
from PIL import Image
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle


page_config = {"page_title": "Human Activity Recognition", "layout": "wide"}
st.set_page_config(**page_config)


st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 1rem;
                    padding-right: 1rem;
                    padding-bottom: 5rem;
                    padding-left: 1rem;
                }
                .stContainer {
                        background-color: navy; /* Navy blue background */
                        border-radius: 10px; /* Rounded corners */
                        padding: 20px; /* Padding inside the boxes */
                        color: white; /* White text color */
                        margin-bottom: 10px; /* Margin between the containers */
                        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); /* Optional: Adds shadow for 3D effect */
                }
        </style>
        """, unsafe_allow_html=True)

###################################################################################
col1, col2, col3 = st.columns([1,4, 1])
with col2:
    st.title("Welcome to HAR Interface")
    st.markdown("![Alt Text](https://media.giphy.com/media/OM87ilEkmKz0zMMKP5/giphy.gif)")


col1, col2, col3 = st.columns(3)
with col1:
    with st.container():
        st.subheader("Introduction")
        st.write("Get a comprehensive introduction into the interface, chosen data, and HAR")

with col2:
    with st.container():
        st.subheader("First-level Analysis")
        st.write("Explore the primary first-level analysis for knowing the data")

with col3:
    with st.container():
        st.subheader("Model Explanation")
        st.write("Understand an in-depth explanation for different models and workings")