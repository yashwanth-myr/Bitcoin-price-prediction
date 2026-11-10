# main_app.py
import streamlit as st
import pandas as pd
import io

# Import the modules of the other files
import prediction
import Bitcoin_price_prediction_randomforest
import svr

st.sidebar.title("Select Option")

# Create a selection box in the sidebar
app_mode = st.sidebar.selectbox(
    "Choose an option",
    ("Upload Data", "XGBoost", "Random Forest", "SVR")
)

required_columns = ['Date', 'Price', 'Vol.', 'Change %']

if app_mode == "Upload Data":
    st.title("Upload Bitcoin Data CSV")
    uploaded_file = st.file_uploader("Upload Bitcoin Data CSV", type=["csv"], key="main_uploader")
    if uploaded_file:
        if uploaded_file.size == 0:
            st.error("Uploaded File is empty!")
        else:
            try:
              df = pd.read_csv(uploaded_file, thousands=',')
              if not all(col in df.columns for col in required_columns):
                  st.error(f"Uploaded file must contain the following columns: {required_columns}")
              else:
                  df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
                  st.session_state['df'] = df
                  st.success("File Uploaded Successfully")
            except Exception as e:
                st.error(f"Error loading the file: {e}")

elif app_mode == "XGBoost":
    if 'df' in st.session_state:
        df = st.session_state['df']
        prediction.run_model(df)  # Pass the DataFrame
    else:
        st.error("Please upload the data first.")
elif app_mode == "Random Forest":
    if 'df' in st.session_state:
        df = st.session_state['df']
        Bitcoin_price_prediction_randomforest.run_model(df)  # Pass the DataFrame
    else:
        st.error("Please upload the data first.")
elif app_mode == "SVR":
    if 'df' in st.session_state:
        df = st.session_state['df']
        svr.run_model(df)  # Pass the DataFrame
    else:
        st.error("Please upload the data first.")