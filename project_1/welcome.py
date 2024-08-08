import streamlit as st

# Set page configuration
st.set_page_config(page_title="National Poll on Healthy Aging", page_icon="🏥", layout="wide")

# Streamlit app function
def app():
   
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>⚕️National Poll on Healthy Aging</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #FF8C00; fontsize:10'>How can health data empower policy makers and insurance providers to optimize decision-making?</h3>", unsafe_allow_html=True)
    
    # Introduction with an image or icon
    st.markdown(
    "<h5 style='text-align: center; margin-bottom: 20px;'>"
    "We aim to provide valuable insights into the health and well-being of older adults, based on the data provided by "
    "Michigan's Institute for Healthcare Policy and Innovation (please refer to 'About the Dataset' page for more information about the dataset). "
    "We hope our app helps policymakers and insurance providers make informed decisions regarding the wellbeing of older adults in the USA."
    "</h5>", unsafe_allow_html=True)

    # Horizontal rule
    st.markdown("<hr style='margin-top: 30px; margin-bottom: 30px;'>", unsafe_allow_html=True)

    # App Features with icons
    st.markdown("<h4 style='text-align: center; margin-bottom: 20px;'>Here is what our app helps you to do:</h4>", unsafe_allow_html=True)
    
    features = [
        {"feature": "Explore the data on National Poll on Healthy Aging", "icon": "📊"},
        {"feature": "Understand some of the basic traits and trends", "icon": "📈"},
        {"feature": "Train a customized Machine Learning Model to predict the number of doctor visits", "icon": "⚙️"},
        {"feature": "See predictions from the model with your own data", "icon": "🔍"},
        {"feature": "See how changing inputs of one parameter affects your model", "icon": "🔄"}
    ]
    
    # List of features as bullet points with icons
    st.markdown("<ul style='font-size:18px;'>", unsafe_allow_html=True)
    for feat in features:
        st.markdown(f"<li style='margin-bottom: 10px;'>{feat['icon']} {feat['feature']}</li>", unsafe_allow_html=True)
    st.markdown("</ul>", unsafe_allow_html=True)

    # Horizontal rule
    st.markdown("<hr style='margin-top: 30px; margin-bottom: 30px;'>", unsafe_allow_html=True)

    # Footer text aligned to bottom-right
    st.markdown(
        "<div style='position: absolute; bottom: 10px; right: 10px; text-align: right; color: #888888;'>Developed by Acharya, De, Ispahani, Saiyed</div>",
        unsafe_allow_html=True
    )

# Execute the app function if this is the main module
if __name__ == '__main__':
    app()
