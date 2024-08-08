import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# color code and plot for 
import plotly.express as px
#sns.set_color_codes(palette = 'deep')
#sns.set_style("darkgrid")
#sns.axes_style("darkgrid")
sns.set_style("darkgrid", {"axes.facecolor": ".9"})


###################################################################################

data = pd.read_csv('pages/aw_fb_data.csv')
data.drop(['Unnamed: 0','X1'],axis=1,inplace=True)
data.rename(columns={'hear_rate': 'heart_rate'}, inplace=True)

# Set the matplotlib style
plt.style.use('classic')
st.set_page_config(layout="wide")

# Sidebar information
st.sidebar.title("About the Dataset")
st.sidebar.markdown("""Target `activity` :""")
activity_counts = data['activity'].value_counts().reset_index()
activity_counts.columns = ['activity', 'count']

# Create pie chart
fig = px.pie(activity_counts, values='count', names='activity')
st.sidebar.plotly_chart(fig, use_container_width=True)

st.title('Exploratory HAR Analysis')

st.write("""
#### This section provides primary descriptive analysis. Use the interactive elements to explore data and visualize distributions.""")
st.markdown("""<hr style="height:0.5px;border:none;color:#333;background-color:#333;" /> """,
        unsafe_allow_html=True)


# Creating two columns
#col1, col2 = st.columns(2)

#with col1:
    # Display data 
    #st.write('## Data Preview')
    #st.write(data.head(8))

#with col2:
# Summary statistics
st.write('### Summary Statistics')
st.write(data.describe())

col3, col4 = st.columns((1,2))

with col4:
    #heatmap correlation
    st.write('### Correlation Heatmap')
    subset = data.drop(['activity','device'],axis=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(subset.corr(), annot=True, fmt='.2f', cmap="crest", ax=ax)
    st.pyplot(fig)

with col3:
    # Check the count of categorical 
    st.write('### Categorical Variables')
    option = st.selectbox(
        'Select a category to display its count:',
        ('gender', 'device', 'activity')
    )
    # Displaying the bar chart for the selected category
    st.write(f'### {option.capitalize()} Count')
    value_counts = data[option].value_counts().reset_index()
    value_counts.columns = [option, 'Counts']
    fig, ax = plt.subplots(figsize=(10, 8))
    bar_plot = sns.barplot(x=option, y='Counts', data=value_counts, palette=["#add8e6", "#4682b4"])  
    st.pyplot(fig)

st.markdown("""<hr style="height:0.5px;border:none;color:#333;background-color:#333;" /> """,
        unsafe_allow_html=True)

#pairplot visualisation
st.write("### Pairplot based on Activity")
pair_col = ['age', 'gender', 'height', 'weight' ,'heart_rate', 'calories', 'distance' ,'activity', 'device']
pair_subset = data[pair_col]
n_samples = 20  
# Perform stratified sampling
pair_df = pair_subset.groupby('activity').apply(lambda x: x.sample(n=n_samples)).reset_index(drop=True)
pairplot_fig = sns.pairplot(pair_df, hue='activity', height=3, palette='colorblind')
st.pyplot(pairplot_fig)

# Function to plot age distribution
def plot_distribution(data, col):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[col], kde=True, bins=30)
    plt.title(f'{col.capitalize()} Distribution')
    plt.xlabel(col.capitalize())
    plt.ylabel('Frequency')
    plt.grid(True) 
    st.pyplot(plt)

    
# Create columns to centre graph
left_d, right_d = st.columns([6, 2])  

with right_d:
    # Check the Distribution of variables 
    st.write('### Distribution')
    option = st.selectbox(
        'Select a category to display its distribution:',
        (data.columns)
    )
with left_d:
    plot_distribution(data, option)

st.markdown("""<hr style="height:0.5px;border:none;color:#333;background-color:#333;" /> """,
        unsafe_allow_html=True)
# Create columns to centre graph
left_r, right_r = st.columns([2, 6]) 

with left_r:
    # Check the Relationship of variables 
    st.write('### Relationship Between Features')
    var1 = st.selectbox(
        'Select Feature 1:',
        (subset.columns)
    )  
    subset2 = subset.drop(var1,axis=1)
    var2 = st.selectbox(
        'Select Feature 2:',
        (subset2.columns)
    )

with right_r:
    fig = px.scatter(data, x=var1 , y=var2, color='activity')
    st.plotly_chart(fig, theme="streamlit")

# # Create columns to centre graph
# left, center2, right = st.columns([1, 6, 1]) 

# with center2:   
#     # Calculating BMI
#     data['bmi'] = data['weight'] / (data['height'] / 100) ** 2
#     st.title('BMI vs Activity Analysis')
#     # Visualize the relationship between BMI and activity
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.boxplot(x='activity', y='bmi', data=data, palette='coolwarm')
#     plt.title('BMI vs Activity')
#     plt.xlabel('Activity')
#     plt.ylabel('BMI')
    
#     # Display the plot in Streamlit
#     st.pyplot(fig)


