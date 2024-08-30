import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import base64

# Set page config at the very beginning
st.set_page_config(page_title="Inotrope Duration Prediction", layout="wide")

# Function to load and encode the image
def get_image_base64(path):
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return f"data:image/png;base64,{encoded}"

# Load the data
@st.cache_data
def load_data():
    sheet_name = "final_streamlit_data"
    dataframe = pd.read_excel(f"{sheet_name}.xlsx")
    return dataframe

dataframe = load_data()

# Define the new feature set and target
new_features = [
    'Age',
    'Sex',
    'BMI',
    'Cardioplegia Volume/Weight (mL/kg)',
    'Cardioplegia - Cardiopulmonary Bypass Time (min)',
    'Cardioplegia - Cross Clamp Time (min)',
    'Cardioplegia - # of Doses',
    'Average Dosing Interval (min)',
]

X_new = dataframe[new_features]
y_new = dataframe['Inotrope > 24 hours']

# Split the dataset into training and testing sets
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

# Retrain the model with the new features
@st.cache_resource
def train_new_model():
    forest_classifier = RandomForestClassifier(random_state=42)
    forest_classifier.fit(X_train_new, y_train_new)
    return forest_classifier

forest_classifier = train_new_model()

# Predictions and evaluation
y_pred_new = forest_classifier.predict(X_test_new)
accuracy_new = accuracy_score(y_test_new, y_pred_new)

# Custom CSS
st.markdown("""
    <style>

    .stButton > button {
        background-color: #c3ab68;
        color: #000000 !important;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover, .stButton > button:active, .stButton > button:focus {
        background-color: #a08d56;
        color: #000000 !important;
    }
    .stNumberInput > div > div > input {
        border-radius: 5px;
    }
    .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #000000;
        padding: 10px 20px;
        border-radius: 10px 10px 0 0;
        margin: -20px -20px 20px -20px;
    }
    .header h1 {
        color: #c3ab68;
        margin: 0;
    }
    .header img {
        width: 70px;
        height: auto;
    }
    .input-container {
        display: flex;
        flex-direction: column;
    }
    .input-container > div {
        width: auto !important;
    }
    .input-container label {
        font-size: 1em;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .input-container input {
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Create three columns with the middle one taking 60% width
left_spacer, center_content, right_spacer = st.columns([2, 6, 2])

# Use the middle column for all content
with center_content:
    # App content goes here
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # App header with logo
    logo_base64 = get_image_base64("buffalo.png")
    st.markdown(
        f"""
        <div class="header">
            <h1>Inotrope Duration Prediction</h1>
            <img src="{logo_base64}">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("""
    This model predicts the likelihood of requiring inotropic support for more than 24 hours postoperatively, 
    based on cardioplegia administration and operative times.
    """)

    # Create two columns for the main content
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Features")
        with st.form(key='input_form'):
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            age = st.number_input("Age", min_value=0, max_value=120, value=50)
            sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
            bmi = st.number_input("BMI", value=21.5)
            total_cardioplegia_volume = st.number_input("Total Cardioplegia Volume (mL)", value=1239)
            weight = st.number_input("Weight (kg)", value=71)
            cpb_time = st.number_input("CPB Time (min)", value=110)
            xc_time = st.number_input("XC Time (min)", value=88)
            doses = st.number_input("Cardioplegia - # of Doses", min_value=1, value=1)
            dosing_interval = st.number_input("Average Dosing Interval (min)", value=0)
            st.markdown('</div>', unsafe_allow_html=True)
            submit_button = st.form_submit_button(label='Predict')

    with col2:
        st.subheader("Prediction")
        if submit_button:
            # Calculated fields
            volume_weight = total_cardioplegia_volume / weight

            user_features = pd.DataFrame({
                'Age': [age],
                'Sex': [sex[1]],  # Use the numeric value for sex
                'BMI': [bmi],
                'Cardioplegia Volume/Weight (mL/kg)': [volume_weight],
                'Cardioplegia - Cardiopulmonary Bypass Time (min)': [cpb_time],
                'Cardioplegia - Cross Clamp Time (min)': [xc_time],
                'Cardioplegia - # of Doses': [doses],
                'Average Dosing Interval (min)': [dosing_interval]
            })

            # Prediction
            prediction = forest_classifier.predict(user_features)
            prediction_proba = forest_classifier.predict_proba(user_features)

            st.markdown(f"<h3>{'Inotrope Support > 24 hours' if prediction[0] else 'Inotrope Support ≤ 24 hours'}</h3>", unsafe_allow_html=True)
            st.markdown(f"<b>Probability of > 24 hours:</b> {prediction_proba[0][1]:.2f}", unsafe_allow_html=True)
            st.markdown(f"<b>Probability of ≤ 24 hours:</b> {prediction_proba[0][0]:.2f}", unsafe_allow_html=True)

            st.subheader("Model Performance")
            st.markdown(f"<b>Accuracy:</b> {accuracy_new:.2f}", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# You can add more sections or visualizations here as needed
