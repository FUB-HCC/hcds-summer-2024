import streamlit as st

title = '<h1 style="font-family:sans-serif; color:RoyalBlue; font-size: 26px; padding-bottom: 0;">Your feedback is valuable</h1>'
st.markdown(title, unsafe_allow_html=True)
st.write("---")
st.text_area("Have questions about usage or suggestions for improvements? We’d love to hear from you", placeholder="Type here...")

st.write("Reach out to us via: ethical_team@fub.de")
st.write("")

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: mediumaquamarine;
    border-color: mediumaquamarine;
}
</style>""", unsafe_allow_html=True)

button = st.button("Submit", type="primary")