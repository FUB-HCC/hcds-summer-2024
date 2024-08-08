import pandas as pd
import plotly.express as px
import streamlit as st

# Load the dataset
@st.cache_data  # Cache the dataset to avoid reloading on every interaction
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Define mappings for categorical features
def apply_mappings(df, mappings):
    for column, mapping in mappings.items():
        if column in df.columns:
            df[column] = df[column].map(mapping)
    return df

# Function to create a percentage plot
def create_percentage_plot(column, filtered_data):
    count_data = filtered_data[column].value_counts(normalize=True).reset_index()
    count_data.columns = [column, 'percentage']
    count_fig = px.bar(count_data, x=column, y='percentage',
                       title=f'Percentage Plot of {column}',
                       text='percentage')
    count_fig.update_layout(yaxis_title='Percentage', xaxis_title=column)
    count_fig.update_yaxes(tickformat="0.1%")
    count_fig.update_xaxes(type='category')
    count_fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    return count_fig

# Function to create a relationship plot
def create_relationship_plot(feature1, feature2, filtered_data):
    relationship_fig = px.histogram(filtered_data, x=feature1, color=feature2, barmode='group',
                                    title=f'Count Plot between {feature1} and {feature2}')
    relationship_fig.update_layout(yaxis_title='Count', xaxis_title=feature1)
    relationship_fig.update_xaxes(type='category')
    return relationship_fig

# Main app function
def app():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>⚕️National Poll on Healthy Aging</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h3>Categorical Features Dashboard</h3>", unsafe_allow_html=True)
    
    # Load the dataset
    file_path = 'NPHA-doctor-visits.csv'  # Replace with your file path
    data = load_data(file_path)
    
    # Define mappings for categorical features
    mappings = {
        'Number of Doctors Visited': {1: '0-1 doctors', 2: '2-3 doctors', 3: '4 or more doctors'},
        'Physical Health': {-1: 'Refused', 1: 'Excellent', 2: 'Very Good', 3: 'Good', 4: 'Fair', 5: 'Poor'},
        'Mental Health': {-1: 'Refused', 1: 'Excellent', 2: 'Very Good', 3: 'Good', 4: 'Fair', 5: 'Poor'},
        'Dental Health': {-1: 'Refused', 1: 'Excellent', 2: 'Very Good', 3: 'Good', 4: 'Fair', 5: 'Poor', 6: 'N/A - I have dentures'},
        'Employment': {-1: 'Refused', 1: 'Working full-time', 2: 'Working part-time', 3: 'Retired', 4: 'Not working at this time'},
        'Stress Keeps Patient from Sleeping': {0: 'No', 1: 'Yes'},
        'Medication Keeps Patient from Sleeping': {0: 'No', 1: 'Yes'},
        'Pain Keeps Patient from Sleeping': {0: 'No', 1: 'Yes'},
        'Bathroom Needs Keeps Patient from Sleeping': {0: 'No', 1: 'Yes'},
        'Unknown Keeps Patient from Sleeping': {0: 'No', 1: 'Yes'},
        'Trouble Sleeping': {-1: 'Refused', 1: '0 nights', 2: '1-2 nights', 3: '3-5 nights'},
        'Prescription Sleep Medication': {-1: 'Refused', 1: 'Use regularly', 2: 'Use occasionally', 3: 'Do not use'},
        'Race': {-2: 'Not asked', -1: 'Refused', 1: 'White, Non-Hispanic', 2: 'Black, Non-Hispanic', 3: 'Other, Non-Hispanic', 4: 'Hispanic', 5: '2+ Races, Non-Hispanic'},
        'Gender': {-2: 'Not asked', -1: 'Refused', 1: 'Male', 2: 'Female'},
        'Age': {1: '50-64 years', 2: '65-80 years'}
    }
    
    
        # Apply mappings
    data = apply_mappings(data, mappings)
    reverse_mappings = {col: {v: k for k, v in mapping.items()} for col, mapping in mappings.items()}

    # Filter section
    st.write("To filter data based on the value of a certain demographic feature, use the 'Enable Filter' checkbox.")
    st.markdown("""
    <style>
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 180px;
        background-color: black;
        color: #fff;
        text-align: center;
        border-radius: 5px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%; /* Position above the icon */
        left: 50%;
        margin-left: -90px; /* Center the tooltip */
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create columns for layout
    col1, col2 = st.columns([1, 10])

    with col1:
        enable_filter = st.checkbox("")
        
    with col2:
        st.markdown("""
            <div style="display: flex; align-items: center;">
                <span style="font-weight: bold; font-size: 16px;">Enable Filter</span>
                <span class="tooltip" style="margin-left: 8px;">
                    <i>&#x1F6C8;</i>
                    <span class="tooltiptext">Analyse a subsection of data, for example apply filter on Females (or Males) if Gender is selected as the demographic feature.</span>
                </span>
            </div>
        """, unsafe_allow_html=True)

    
    if enable_filter:
        col1, col2 = st.columns(2)
        with col1:
            filter_feature = st.selectbox("Select the Demographic Feature:", options=['Gender', 'Race', 'Age', 'Employment'])
            #filter_feature = st.selectbox('Select the feature to filter data:', data.columns)
        with col2:
            filter_value = st.selectbox(f'Select the value for {filter_feature}:', data[filter_feature].unique())
        filtered_data = data[data[filter_feature] == filter_value]
        st.markdown(f'The dataset is now filtered using the "{filter_feature}" category value, "{filter_value}"')
        st.markdown("<br>", unsafe_allow_html=True)  # Add blank space
        st.write("#### Relationship between features based on filtered data")
    else:
        filtered_data = data
        st.markdown("<br>", unsafe_allow_html=True)  # Add blank space
        st.write("#### Relationship between features based on the original data")
    
    # Relationship between two features
        # Dropdowns for selecting features to visualize
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    col1, col2 = st.columns(2)
    with col1:
       x_axis = st.selectbox('Select the X-axis feature (option1):', options=categorical_cols, key='x_axis')
    with col2:
       y_axis = st.selectbox('Select the Y-axis feature for the count (option2):', options=categorical_cols, key='y_axis')
    # feature = st.selectbox('Select the feature:', data.columns, index=0)
    st.plotly_chart(create_percentage_plot(x_axis, filtered_data))
    st.markdown(f"##### *Interpretation*")
    st.markdown(f'The above bar plot shows the percentage of population belonging to each category for the feature "{x_axis}".')
    st.plotly_chart(create_relationship_plot(x_axis, y_axis, filtered_data))
    st.markdown(f"##### *Interpretation*")
    st.markdown(f'The above count plot shows the distribution of "{y_axis}" across different categories of "{x_axis}".')
    

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("Based on these distribution charts, the policy makers can easily visualise the relationships and check whether there is uniformity, dominance or absence of data among various categories. This can help to further determine which group in particular requires more attention.")
# Run the app
if __name__ == '__main__':
    app()
