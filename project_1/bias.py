import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import RandomOverSampler

def app():
    # Load the data
    data = pd.read_csv('NPHA-doctor-visits.csv')

    # Mappings for better readability in pie charts
    mappings = {
        'Race': {
            -2: 'Not asked', -1: 'Refused', 1: 'White, Non-Hispanic', 2: 'Black, Non-Hispanic', 
            3: 'Other, Non-Hispanic', 4: 'Hispanic', 5: '2+ Races, Non-Hispanic'
        },
        'Employment': {
            -1: 'Refused', 1: 'Working full-time', 2: 'Working part-time', 3: 'Retired', 
            4: 'Not working at this time'
        }
    }

    # Separate 'Race' and 'Employment' columns for resampling
    race_data = data[['Race']]
    employment_data = data[['Employment']]

    # Apply RandomOverSampler to each column separately
    oversampler = RandomOverSampler(random_state=42)
    resampled_race, _ = oversampler.fit_resample(race_data, race_data['Race'])
    resampled_employment, _ = oversampler.fit_resample(employment_data, employment_data['Employment'])

    # Ensure the number of samples is the same for both resampled datasets
    min_samples = min(len(resampled_race), len(resampled_employment))
    resampled_race = resampled_race.sample(n=min_samples, random_state=42)
    resampled_employment = resampled_employment.sample(n=min_samples, random_state=42)

    # Combine resampled data into a single DataFrame
    combined_df = pd.concat([resampled_race.reset_index(drop=True), resampled_employment.reset_index(drop=True)], axis=1)

    # Function to plot pie charts with mapped labels
    def plot_pie_chart(data, column, title):
        labels, counts = np.unique(data[column], return_counts=True)
        mapped_labels = [mappings[column][label] for label in labels]
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.pie(counts, labels=mapped_labels, autopct='%1.1f%%', startangle=140, colors=colors)
        ax.set_title(title)
        ax.axis('equal')
        st.pyplot(fig)

    # Streamlit app
    st.title("Data Distribution Viewer")
    
    st.write("Using Chi-Square Statistic, we checked that out of the demographic features, Race and Employment have significantly uneven distribution. Therefore, we tried to introduce some Random Over Sampling to ensure data distribution is equal for all categories.")

    # Checkbox to switch between original and resampled data
    use_resampled = st.checkbox("Show Resampled Data", value=False)

    if use_resampled:
        st.write("### Resampled Data Distributions")
        plot_pie_chart(combined_df, 'Race', title='Race Distribution (After Resampling)')
        plot_pie_chart(combined_df, 'Employment', title='Employment Distribution (After Resampling)')
    else:
        st.write("### Original Data Distributions")
        plot_pie_chart(data, 'Race', title='Race Distribution (Before Resampling)')
        plot_pie_chart(data, 'Employment', title='Employment Distribution (Before Resampling)')
        
    st.write('### As a future scope of this project, we would like to use this synthetically generated dataset with evenly distributed classes for training the prediction model and analyze how the model performance is affected.')


if __name__ == '__main__':
    app()

