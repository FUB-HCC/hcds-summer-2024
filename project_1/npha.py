import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'NPHA-doctor-visits.csv'
df = pd.read_csv(file_path)
# st.set_page_config(page_title="⚕️National Poll on Healthy Aging", page_icon="🏥", layout="wide")

# Function to display dataset summary with added links
def display_dataset_summary():

    st.markdown("<h3 style='text-align: left; '>📄 Dataset Summary</h3>", unsafe_allow_html=True)
    st.markdown("""
        The University of Michigan National Poll on Healthy Aging (NPHA) gathers information on
        health, health care, and health policy issues impacting Americans 50 years of age and older for the
        general public, healthcare providers, policymakers, and advocates by utilizing the viewpoints of older
        adults and their carers.
        
        The survey is intended to be a regular, nationally representative household survey of adult
        Americans, allowing rapid assessment of issues.
        
        For further exploration and usage of the dataset, refer to the following links:
        - [UCI Repository link: National Poll on Healthy Aging Dataset](https://archive.ics.uci.edu/dataset/936/national+poll+on+healthy+aging+(npha))
        - [ICPSR repository link: National Poll on Healthy Aging (NPHA) Dataset](https://www.icpsr.umich.edu/web/NACDA/studies/37038)
    """)

# Function to display dataset information with enhanced aesthetics
def display_dataset_info(df):
    with st.expander("**More Information**"):
        st.markdown("""
            - **Created by:** Preeti N. Malani, Jeffrey Kullgren, and Erica Solway from the University of Michigan
            - **Date of Creation:** April 2017
            - **Funded by:** AARP and Michigan Medicine, the University of Michigan's academic medical center
            - **Purpose:** To investigate health, healthcare, and health policy issues affecting Americans aged 50 and older
            - **Collection Method:** Conducted by GfK Group (formerly Knowledge Networks) using a probability-based web panel
            - **Sampling Method:** Representative sample of US citizens aged 50-80, non-institutionalized
            - **Excluded Data Points:** Participants with missing responses for health and sleep features
            - **Processing:** Selection of 14 sleep and health-related features, elimination of respondents with missing values
            - **Completion Rate:** 75% for ages 50-64, 80% for ages 65-80
            - **Attribute Clarity:** Each attribute is clearly defined with easy-to-understand naming conventions
            - **Protected Attributes:** Race, gender, age, employment and health-related information
        """)
        st.markdown("##### 📚 **Dataset Statistics**")
        st.write(f"**Number of rows (Patients):** {df.shape[0]}")
        st.write(f"**Number of columns (Characterstics):** {df.shape[1]}")
        st.write("**There are no missing values present in the dataset**")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("**Key Features / Attributes**"):
        st.markdown("""
            The data is presented in a tabular (.csv) format; key features include:
            - **Number_of_Doctors_Visited:** Target variable indicating the count of different doctors seen in a year (1: 0-1 doctors, 2: 2-3 doctors, 3: 4 or more doctors)
            - **Age:** Categorical (1: 50-64 years, 2: 65-80 years)
            - **Physical_Health:** Categorical (1: Excellent, 2: Very Good, 3: Good, 4: Fair, 5: Poor, -1: Refused)
            - **Mental_Health:** Categorical (1: Excellent, 2: Very Good, 3: Good, 4: Fair, 5: Poor, -1: Refused)
            - **Dental_Health:** Categorical (1: Excellent, 2: Very Good, 3: Good, 4: Fair, 5: Poor, -1: Refused)
            - **Employment:** Categorical (1: Working full-time, 2: Working part-time, 3: Retired, 4: Not working at this time, -1: Refused)
            - **Stress_Keeps_Patient_from_Sleeping:** Binary (0: No, 1: Yes)
            - **Medication_Keeps_Patient_from_Sleeping:** Binary (0: No, 1: Yes)
            - **Pain_Keeps_Patient_from_Sleeping:** Binary (0: No, 1: Yes)
            - **Bathroom_Needs_Keeps_Patient_from_Sleeping:** Binary (0: No, 1: Yes)
            - **Unknown_Keeps_Patient_from_Sleeping:** Binary (0: No, 1: Yes)
            - **Trouble_Sleeping:** Binary (0: No, 1: Yes)
            - **Prescription_Sleep_Medication:** Categorical (1: Use regularly, 2: Use occasionally, 3: Do not use, -1: Refused)
            - **Race:** Categorical (1: White, Non-Hispanic, 2: Black, Non-Hispanic, 3: Other, Non-Hispanic, 4: Hispanic, 5: 2+ Races, Non-Hispanic, -1: Refused, -2: Not asked)
        """)

def plot_pie_charts(df):
    st.markdown("<h3 style='text-align: left;'>📈 Distribution of Different Demographic Factors</h3>", unsafe_allow_html=True)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Gender Distribution
    gender_labels = {1: "1: Male", 2: "2: Female", -1: "-1: Refused", -2: "-2: Not asked"}
    gender_counts = df['Gender'].map(gender_labels).value_counts()
    wedges, texts, autotexts = axs[0].pie(gender_counts, autopct='%1.1f%%', colors=['#FF9999', '#66B3FF', '#99FF99', '#FFCC99'],
                                          startangle=140, pctdistance=1.15, labeldistance=1.2, shadow=False)
    axs[0].set_title('Gender Distribution', fontsize=16)
    axs[0].legend(wedges, gender_counts.index, title="Gender", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=12)
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_verticalalignment('center')  # Ensures text alignment

    # Race Distribution
    race_labels = {1: "1: White, Non-Hispanic", 2: "2: Black, Non-Hispanic", 3: "3: Other, Non-Hispanic", 4: "4: Hispanic", 5: "5: 2+ Races, Non-Hispanic", -1: "-1: REFUSED", -2: "-2: Not asked"}
    race_counts = df['Race'].map(race_labels).value_counts()
    wedges, texts, autotexts = axs[1].pie(race_counts, autopct='%1.1f%%', colors=['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#FF6666', '#CCCCFF', '#FFFF99'],
                                          startangle=140, pctdistance=1.15, labeldistance=1.2, shadow=False)
    axs[1].set_title('Race Distribution', fontsize=16)
    axs[1].legend(wedges, race_counts.index, title="Race", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=12)
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_verticalalignment('center')  # Ensures text alignment

    # Employment Distribution
    employment_labels = {1: "1: Working full-time", 2: "2: Working part-time", 3: "3: Retired", 4: "4: Not working at this time", -1: "-1: Refused"}
    employment_counts = df['Employment'].map(employment_labels).value_counts()
    wedges, texts, autotexts = axs[2].pie(employment_counts, autopct='%1.1f%%', colors=['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#FF6666'],
                                          startangle=140, pctdistance=1.15, labeldistance=1.2, shadow=False)
    axs[2].set_title('Employment Distribution', fontsize=16)
    axs[2].legend(wedges, employment_counts.index, title="Employment", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=12)
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_verticalalignment('center')  # Ensures text alignment

    plt.tight_layout()
    st.pyplot(fig)


    st.markdown(f"##### *Interpretation*")
    st.markdown("""
        - **Gender Distribution:** The pie chart shows the proportion of different genders in the dataset.
        - **Race Distribution:** This chart depicts the racial composition of the respondents. (1: White, Non-Hispanic, 2: Black, Non-Hispanic, 3: Other, Non-Hispanic, 4: Hispanic, 5: 2+ Races, Non-Hispanic, -1: REFUSED, -2: Not asked)
        - **Employment Distribution:** This chart illustrates the employment status of the respondents. (1: Working full-time, 2: Working part-time, 3: Retired, 4: Not working at this time, -1: Refused)
    """)


def plot_health_status(df, demographic_feature):
    feature_mapping = {
        'Race': 'Race',
        'Gender': 'Gender',
        'Age': 'Age',
        'Employment': 'Employment'
    }

    label_mapping = {
        'Race': {1: "White, Non-Hispanic", 2: "Black, Non-Hispanic", 3: "Other, Non-Hispanic", 4: "Hispanic", 5: "2+ Races, Non-Hispanic", -1: "REFUSED", -2: "Not asked"},
        'Gender': {1: "Male", 2: "Female", -1: "REFUSED", -2: "Not asked"},
        'Age': {1: "50-64 years", 2: "65-80 years"},
        'Employment': {1: "Working full-time", 2: "Working part-time", 3: "Retired", 4: "Not working at this time", -1: "Refused"}
    }

    df_filtered = df[df[demographic_feature] >= 0]  # Filter out negative values
    df_grouped = df_filtered.groupby(demographic_feature).mean()
    df_grouped.index = df_grouped.index.map(label_mapping[demographic_feature])

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.25
    index = range(len(df_grouped))

    bar1 = ax.barh(index, df_grouped['Physical Health'], bar_width, label='Avg Physical Health', color='#FF9999')
    bar2 = ax.barh([i + bar_width for i in index], df_grouped['Mental Health'], bar_width, label='Avg Mental Health', color='#66B3FF')
    bar3 = ax.barh([i + 2 * bar_width for i in index], df_grouped['Dental Health'], bar_width, label='Avg Dental Health', color='#99FF99')

    # Display numeric values on bars
    for i, (ph, mh, dh) in enumerate(zip(df_grouped['Physical Health'], df_grouped['Mental Health'], df_grouped['Dental Health'])):
        ax.text(ph + 0.1, i, f'{ph:.2f}', va='center', fontsize=10, color='black')
        ax.text(mh + 0.1, i + bar_width, f'{mh:.2f}', va='center', fontsize=10, color='black')
        ax.text(dh + 0.1, i + 2 * bar_width, f'{dh:.2f}', va='center', fontsize=10, color='black')

    ax.set_ylabel(feature_mapping[demographic_feature])
    ax.set_xlabel('Average Health Rating')
    ax.set_title(f'Status of Overall Health based on {feature_mapping[demographic_feature]}')
    ax.set_yticks([i + bar_width for i in index])
    ax.set_yticklabels(df_grouped.index)
    
    # Set x-axis limits and ticks
    ax.set_xlim(0, 5)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(['0', '1: Excellent', '2: Very Good', '3: Good', '4: Fair', '5: Poor'])

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Place legend outside the graph
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    st.pyplot(fig)



    # Interpretation
    st.markdown(f"##### *Interpretation*")
    st.markdown(f"The bar chart above presents the average physical, mental, and dental health ratings categorized by different demographic factors: Race, Gender, Age, and Employment. Here are some key insights:")
    st.markdown(f"1. *Average Physical Health*: This bar (colored in light red) shows the average physical health rating across different categories of the selected demographic factor. For example, if 'Gender' is selected, it compares the average physical health rating of males and females.")
    st.markdown(f"2. *Average Mental Health*: This bar (colored in light blue) represents the average mental health rating for the selected demographic categories. It allows us to compare how different groups perceive their mental health.")
    st.markdown(f"3. *Average Dental Health*: The green bar indicates the average dental health rating across the categories. This can highlight disparities in dental health among different groups.")
    st.markdown(f"By analyzing these averages, policymakers and insurance providers can identify which demographic groups might require more attention or resources. For example, if one demographic group consistently reports lower health ratings, targeted interventions can be designed to address those specific needs.")

# Streamlit app
def app():
    # Header
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>⚕️National Poll on Healthy Aging</h1>", unsafe_allow_html=True)
#     st.markdown("<h4 style='text-align: center; color: #FF8C00; fontsize:10'>How can health data empower policy makers and insurance providers to optimize decision-making?</h2>", unsafe_allow_html=True)

    st.markdown("---")

    # Dataset summary section
    #st.markdown("<div style='background-color: #f9f9f9; padding: 15px; border-radius: 10px;'>", unsafe_allow_html=True)
    display_dataset_summary()
    st.markdown("</div>", unsafe_allow_html=True)

    # More information dropdown
    display_dataset_info(df)

    # Dataset snippet section
    st.markdown("<h3 style='text-align: left; '>🔍 Dataset Snippet</h3>", unsafe_allow_html=True)
    st.markdown("""
        Most of the attributes in the dataset are **categorical** such as Mental Health, Physical Health, and Dental Health. Each attribute is rated on a scale from 1 to 5, where 1 indicates excellent health and 5 indicates poor health. Detailed descriptions of each attribute can be found under the [Key Features](#expander-content-Key_Features) dropdown.
    """)
    df.index.name = 'Subject'
    st.dataframe(df.head(6))

    # Distribution of demographic factors
    plot_pie_charts(df)

    # Dropdown for demographic features
    st.markdown("<h3 style='text-align: left; '>📊 Status of Overall Health based on Different Demographic Factors</h3>", unsafe_allow_html=True)
    demographic_feature = st.selectbox("##### **Select Demographic Feature from the dropdown menu below:**", options=['Race', 'Gender', 'Age', 'Employment'])
    plot_health_status(df, demographic_feature)

# Execute the app function if this is the main module
if __name__ == '__main__':
    app()
