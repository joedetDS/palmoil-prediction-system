import streamlit as st
import pickle
import numpy as np

# Load the Random Forest model
with open('random_forest_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# Streamlit App
st.title("Oil Quality Prediction")
st.markdown("""
This app predicts whether an oil sample is **Pure** or **Adulterated** based on its chemical properties.
Provide the values for the following parameters:
""")

# Arrange input fields side by side
col1, col2 = st.columns(2)

with col1:
    oleic_acid = st.slider("Oleic Acid (%)", min_value=13.62, max_value=73.23, value=44.92, step=0.01)
    peroxide_value = st.slider("Peroxide Value (meq/kg)", min_value=1.00, max_value=14.99, value=8.04, step=0.01)
    free_fatty_acid = st.slider("Free Fatty Acid (%)", min_value=0.10, max_value=5.00, value=2.61, step=0.01)

with col2:
    iodine_value = st.slider("Iodine Value (g/100g)", min_value=50.00, max_value=69.99, value=59.98, step=0.01)
    saponification_value = st.slider("Saponification Value (mg KOH/g)", min_value=190.00, max_value=209.99, value=199.89, step=0.01)
    viscosity = st.slider("Viscosity (cP)", min_value=30.01, max_value=99.99, value=64.72, step=0.01)

# Predict button
if st.button("Predict"):
    # Create a feature array
    features = np.array([[oleic_acid, peroxide_value, free_fatty_acid, iodine_value, saponification_value, viscosity]])

    # Make prediction
    prediction = rf_model.predict(features)[0]  # 0 for the first prediction

    # Map prediction to class
    prediction_class = "Adulterated" if prediction == 0 else "Pure"

    # Display result
    st.write(f"The oil sample is predicted to be: **{prediction_class}**")

# Add an informational sidebar
st.sidebar.title("About")
st.sidebar.info("This app uses a trained Random Forest model to classify oil samples based on their chemical properties.")
st.sidebar.image("model_comparison.png", caption="Model Comparison: SVM vs RF vs LR", use_column_width=True)
