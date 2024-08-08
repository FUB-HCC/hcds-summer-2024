# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 17:36:46 2024

@author: sk
"""

import streamlit as st
from PIL import Image

#from home_page import show_home_page
from streamlit_option_menu import option_menu
from PIL import Image
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

##################################################################################
# st.markdown("""
#         <style>
#                .css-18e3th9 {
#                     padding-top: 0rem;
#                     padding-bottom: 10rem;
#                     padding-left: 5rem;
#                     padding-right: 5rem;
#                 }
#                .css-1d391kg {
#                     padding-top: 1rem;
#                     padding-right: 1rem;
#                     padding-bottom: 3.5rem;
#                     padding-left: 1rem;
#                 }
#         </style>
#         """, unsafe_allow_html=True)
# ###################################################################################


#img_logo = Image.open("logo.jpg")
#page_config = {"page_title": "Human Activity Recognition", "layout": "centered"}
#st.set_page_config(**page_config)

page_2 = option_menu(
    menu_title=None,
    options=["Dashboard", "Data", "HAR"],
    #icons=["motherboard", "file-earmark-code","house-fill"],
    default_index=0,
    orientation="horizontal",
    styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "black", "font-size": "16px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "red"}
            }
)

def show_Data_page():
    """
    This function displays the DATA
    """
    st.title("Data Description")
    
    
    #ol1, col2, col3 = st.columns([1, 3, 1])
    #with col2:
     #   img = Image.open("HAR_2.png")
      #  st.image(img)
    #st.caption("### **Open Dataset**")  
    
    st.caption("### **Purpose of the research is to examine whether machine learning models can predict six different  movement types from sensor data stored with commercial wearables.**")
    st.markdown("Open dataset : https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZS2Z2J")
    
    
    st.markdown("This preprocessed dataset includes 3656 minutes of Apple Watch data and 2608 minutes of Fitbit data.""") 
    st.markdown("* **Y** : Target outcome variables are six activity classes - lying, sitting, walking self-paced, 3 METS, 5 METS, and 7 METS  (different intensities of running).")
    st.markdown("* **X** : Minute-by-minute heart rate, steps, distance, and calories from two wearables - Apple Watch and Fitbit. Indirect calorimetry used to measure energy expenditure.")  
    st.markdown("""<hr style="height:0.5px;border:none;color:#333;background-color:#333;" /> """,
                unsafe_allow_html=True)
    
    
    
    st.markdown("""Summary of Participants:""")
    st.markdown("* Participants recruited using social media post and word-of-mouth in Canada.") 
    st.markdown("* Inclusion Criterion - 18 years and above,  completing the Physical Activity Readiness Questionnaire.")
    st.markdown("* 26 women(0) and 20 men(1) wearing : Apple Watch and Fitbit Charge HR2.")
    st.markdown("* Participants complete 40-minutes of total treadmill time and 25-minutes of sitting or lying time.")
    st.markdown("* Participants provided signed informed consent and didnot recieve any compensation.")
    
    st.markdown("**Paients or public were not involved in the design, conduct, or reporting, or dissemination plans of this research.**")

    st.markdown("""<hr style="height:0.5px;border:none;color:#333;background-color:#333;" /> """,
            unsafe_allow_html=True)

    #with col1:
        # Display data 
    data = pd.read_csv('pages/aw_fb_data.csv')
    data.drop(['Unnamed: 0','X1'],axis=1,inplace=True)
    data.rename(columns={'hear_rate': 'heart_rate'}, inplace=True)
    st.write('### Data Preview')
    st.write(data.head(8))

   
    
def show_Dashboard() :  
    
    st.title(" HAR Explanation Interface")
    #st.caption("### ***")
    st.caption("### **Goal: Can ML predict phsical activity from wearable data? Are the predictions reproducible, interpretable and scalable?** ")

    st.caption("""#### Challenge :  
               Wearables-based human activity recognition research still relies on very small datasets, the majority collected in artificial settings.Further, this small-data limitation confounds findings involving data-hungry deep learning methods""")
   
    st.markdown("""<hr style="height:0.25px;border:none;color:#333;background-color:#333;" /> """,
                unsafe_allow_html=True)
    
    st.caption("## **HAR explanation interface will enable the user to understand the following,**")
    
    st.markdown("""* **Can ML techniques be deployed to predict activity wearable data ?**""")
                      
    st.write("""* **What is the chosen data and properties ?**""")
    st.write("""* **What are the performance metrics ?**""")
    st.write("""* **Are the predictions fair,  reproducible and scalable ?**""")
    st.markdown("""* **How do the outcomes vary for different models ?**""")
    st.markdown("""* **How do the outcomes vary across wearable brands ?**""")
    st.markdown("""* **How do the model perform with reduced features ?**""")
    st.markdown("""* **Which are the protected attributes accounted for in the model ?**""") 
    st.markdown("""* **Which features are deemed useful for the model ?**""")
    st.markdown("""* **What if - features or data are perturbed impacts the resulting model output ?**""") 

    ##########################################################
    
    

#########################################################################


    
def show_HAR_page():
    """
    This function displays the Home Page.
    """
    st.title("Human Activity Recognition (HAR)")
    st.text("")
    # col1, col2, col3 = st.columns([1, 3, 1])
    col1, col2, col3, col4, col5 = st.columns([1, 1, 3, 1, 1])
    #st.text("")
    #st.text("")
    st.caption("## **Context**")
    st.markdown("""
                Human activity recognition (HAR) is an emerging research area, especially  ubiquitous for human behavior analysis and human-computer interaction in real-time. As the name suggests, HAR involves employing machine learning algorithms to recognise simple and complex activities such as walking, running, cooking, factory worker etc. based on different  multimodal data generated by a variety of sensors.  Ability to correctly and timely identify activities is deemed useful with several daily use cases from maintaining healthy lifestyle to providing personalized patient rehabilitation or diagnosing an ongoing illness or pre-symptomatic detection of Covid-19 from wearable data.""")  


    st.caption("## **Motivation for using Wearable  dataset**""")

    st.markdown("""Wearable devices have emerged as tools for measuring detailed physical activity. Moreover, these devices store data for Behaviour in-the-wild (real-life). Therefore, wearables are seen as a potential tool for Behavioral context recognition. This allows us to readily monitor and provide real-time feedback based on activity measurements. Alongside,  the ever-advancing machine learning tools can now help us gauge more fine-tuned predictions and deeper interpretations for movement classification at population level.""")
             
    
    # with col3:
    #     img = Image.open(r"C:\Users\sk\Desktop\HCDS\Project_HCDS\HAR_2.png")
    #     st.image(img)
    #     st.markdown("""<hr style="height:0.00px;border:none;color:#333;background-color:#333;" /> """,
    #                 unsafe_allow_html=True)
    #     # st.markdown("### Human Activity Recognition")
    #     #st.text("")
    #     #st.text("")
    #     #st.markdown("![Alt Text](https://media.giphy.com/media/OM87ilEkmKz0zMMKP5/giphy.gif)")
      


     
    
 
    
if page_2 == "HAR":
    show_HAR_page()

    
if page_2 == "Data":
    show_Data_page()
    

if page_2 == "Dashboard":
    show_Dashboard()

  
  
###################################################################################    
#################################################################
# with st.sidebar :
#     #width: 375px
#     page = option_menu(
#         menu_title=None,
#         options=["Home","Explore", "Analysis", "Summary"],
#         icons=["house-fill", "book", "motherboard", "gear"],
#         default_index=0,
#         orientation="horizontal",
#         styles={
#             "container": {"padding": "0!important", "background-color": "#fafafa"},
#             "icon": {"color": "black", "font-size": "12px"},
#             "nav-link": {
#                 "font-size": "12px",
#                 "text-align": "center",
#                 "margin": "0px",
#                 "--hover-color": "#eee",
#             },
#             "nav-link-selected": {"background-color": "grey"}
#             }
# )


# if page == "Explore":
#     st.title("# Know our data")
#     st.markdown(
#       """
#       Using the left filters, you can perfrom exploratory analsyis """)
 
#     import CDC_data
#     #show_home_page()



# if page == "Analysis":
#     st.write('SAMPLE_3')
#     #show_home_page()


# if page == "Summary":
#     st.write('SAMPLE_4')
#     #show_home_page()



