import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

model = xgb.XGBRegressor()
model.load_model('models/food_delivery_time_model.json')

st.title('🍔 Food Delivery Time Predictor')
st.write('Fill in the details below to predict delivery time')

# correlation data from your actual results
correlation_data = {
    'multiple_deliveries': 0.379,
    'road_traffic_density_Jam': 0.351,
    'delivery_person_age': 0.303,
    'festival_Yes': 0.289,
    'order_hour': 0.184,
    'vehicle_motorcycle': 0.165,
    'city_Semi-Urban': 0.149,
    'weather_Fog': 0.128,
    'distance': 0.096,
    'delivery_person_ratings': -0.360,
    'road_traffic_density_Low': -0.387,
    'vehicle_condition': -0.243,
    'weather_Sunny': -0.209,
    'festival_No': -0.207,
    'city_Urban': -0.188,
}

predictor_col, chart_col = st.columns([1.5, 1])

with predictor_col:
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider('Delivery Person Age', 18, 50, 25)
        ratings = st.slider('Delivery Person Rating', 1.0, 5.0, 4.5, step=0.1)
        vehicle_condition = st.selectbox('Vehicle Condition', [0, 1, 2, 3],
                                          format_func=lambda x: ['Poor','Fair','Good','Excellent'][x])
        multiple_deliveries = st.selectbox('Multiple Deliveries', [0, 1, 2, 3])
        distance = st.slider('Distance', 0.14, 0.51, 0.35, step=0.01)

    with col2:
        weather = st.selectbox('Weather', ['Cloudy', 'Fog', 'Sandstorms', 'Stormy', 'Sunny', 'Windy'])
        traffic = st.selectbox('Traffic Density', ['Low', 'Medium', 'High', 'Jam'])
        order_type = st.selectbox('Type of Order', ['Buffet', 'Drinks', 'Meal', 'Snack'])
        vehicle_type = st.selectbox('Vehicle Type', ['bicycle', 'electric_scooter', 'motorcycle', 'scooter'])
        festival = st.selectbox('Festival', ['No', 'Yes'])
        city = st.selectbox('City Type', ['Urban', 'Semi-Urban', 'Metropolitian'])
        order_hour = st.slider('Order Hour', 0, 23, 19)
        cook_diff = st.slider('Prep Time (minutes)', 5, 15, 10)

    def make_input():
        data = {
            'delivery_person_age': age,
            'delivery_person_ratings': ratings,
            'vehicle_condition': vehicle_condition,
            'multiple_deliveries': multiple_deliveries,
            'distance': distance,
            'cook_diff': cook_diff,
            'order_hour': order_hour,
            'weatherconditions_conditions Fog': 1 if weather == 'Fog' else 0,
            'weatherconditions_conditions Sandstorms': 1 if weather == 'Sandstorms' else 0,
            'weatherconditions_conditions Stormy': 1 if weather == 'Stormy' else 0,
            'weatherconditions_conditions Sunny': 1 if weather == 'Sunny' else 0,
            'weatherconditions_conditions Windy': 1 if weather == 'Windy' else 0,
            'road_traffic_density_Jam ': 1 if traffic == 'Jam' else 0,
            'road_traffic_density_Low ': 1 if traffic == 'Low' else 0,
            'road_traffic_density_Medium ': 1 if traffic == 'Medium' else 0,
            'type_of_order_Drinks ': 1 if order_type == 'Drinks' else 0,
            'type_of_order_Meal ': 1 if order_type == 'Meal' else 0,
            'type_of_order_Snack ': 1 if order_type == 'Snack' else 0,
            'type_of_vehicle_motorcycle ': 1 if vehicle_type == 'motorcycle' else 0,
            'type_of_vehicle_scooter ': 1 if vehicle_type == 'scooter' else 0,
            'festival_No ': 1 if festival == 'No' else 0,
            'festival_Yes ': 1 if festival == 'Yes' else 0,
            'city_NaN ': 0,
            'city_Semi-Urban ': 1 if city == 'Semi-Urban' else 0,
            'city_Urban ': 1 if city == 'Urban' else 0,
        }
        return pd.DataFrame([data])

    if st.button('Predict Delivery Time'):
        input_df = make_input()
        prediction = model.predict(input_df)[0]
        st.success(f'🕐 Estimated Delivery Time: **{prediction:.0f} minutes**')
        st.write('---')
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Time", f"{prediction:.0f} min")
        m2.metric("Traffic", traffic)
        m3.metric("Weather", weather)

with chart_col:
    st.write('### What affects delivery time?')
    st.write('Positive = increases time, Negative = decreases time')

    corr_df = pd.DataFrame({
        'feature': list(correlation_data.keys()),
        'correlation': list(correlation_data.values())
    }).sort_values('correlation')

    fig, ax = plt.subplots(figsize=(4, 6))
    colors = ['#F0997B' if x > 0 else '#5DCAA5' for x in corr_df['correlation']]
    ax.barh(corr_df['feature'], corr_df['correlation'], color=colors)
    ax.axvline(x=0, color='gray', linewidth=0.8)
    ax.set_xlabel('Correlation with delivery time')
    ax.set_title('Feature Correlation')
    plt.tight_layout()
    st.pyplot(fig)

    st.write('🔴 Red = makes delivery slower')
    st.write('🟢 Green = makes delivery faster')