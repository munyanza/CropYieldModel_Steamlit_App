import streamlit as st 
import pandas as pd
import joblib

st.title("Crop Yield Prediction App")

model = joblib.load("model.pkl")
pre = joblib.load("preprocessor.pkl")

# 'rainfall_mm', 'soil_quality_index', 'farm_size_hectares','sunlight_hours', 'fertilizer_kg'
Rainfall = st.text_input("rainfall_mm","0")
Soil_Quality = st.text_input("soil_quality_index","0")
Farm_size_Hectares = st.text_input("farm_size_hectares","0")
Sunlight_Hours = st.text_input("sunlight_hours","0")
Fertilizer_KG = st.text_input("fertilizer_kg","0")

if st.button("Predict"):
    df_input = pd.DataFrame([{
        'rainfall_mm':Rainfall,
        'soil_quality_index':Soil_Quality, 
        'farm_size_hectares':Farm_size_Hectares,
        'sunlight_hours':Sunlight_Hours, 
        'fertilizer_kg':Fertilizer_KG

    }])

    Xp = pre.transform(df_input)
    pred = model.predict(Xp)[0]

    st.success(f"Crop Yield is {pred:.2f} tons")