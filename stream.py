import pandas as pd
import joblib
import streamlit as st
model=joblib.load("chanceadmit.pkl")
st.title("Graduate Admission Chances")
st.write("Enter the Details below:")
GRE_Score = st.number_input("GRE Score")
TOEFL_Score = st.number_input("TOEFL Score")
University_Rating = st.number_input("University Rating")
SOP = st.number_input("SOP")
LOR = st.number_input("LOR")
GPA = st.number_input("GPA")
Research = st.number_input("Research")

# Create dataframe using SAME column names as training
input_df = pd.DataFrame({
    'GRE Score': [GRE_Score],
    'TOEFL Score': [TOEFL_Score],
    'University Rating': [University_Rating],
    'SOP': [SOP],
    'LOR ': [LOR],     # check space!
    'GPA': [GPA],
    'Research': [Research]
})

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"Chance of Admission: {prediction}")
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

st.subheader("Admission Prediction Result")

probability = prediction

st.metric(label="Predicted Chance of Admission",
          value=f"{round(probability * 100, 2)}%")

if probability >= 0.8:
    st.success("Strong chance of admission! 🔥")
elif probability >= 0.6:
    st.warning("Moderate chance. Improve profile slightly.")
else:
    st.error("Low chance. Major improvements needed.")    

chart_data = pd.DataFrame({
    "Category": ["Admission Chance", "Risk"],
    "Value": [probability, 1 - probability]
})

st.bar_chart(chart_data.set_index("Category"))
feature_names = ["GRE_Score", "TOEFL_Score", "University_Rating",
                 "SOP", "LOR", "GPA", "Research"]

coefficients = model.coef_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Impact": coefficients
})

st.subheader("Feature Impact Analysis")
st.bar_chart(importance_df.set_index("Feature"))
st.subheader("What If You Improve GRE by 10 Points?")
import numpy as np

input_data = np.array([[GRE_Score, TOEFL_Score, University_Rating, SOP, LOR , GPA, Research]])

improved_input = input_data.copy()
improved_input[0][0] += 10  # assuming GRE is first column

new_prediction = model.predict(improved_input)[0]

st.write("New Chance:",
         f"{round(new_prediction * 100, 2)}%")

