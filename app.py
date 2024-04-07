import streamlit as st
import pickle 
import numpy as np
import pandas as pd

# Load the model and dataframe
data = pd.read_csv('laptoop.csv')
pipe = pickle.load(open("pipe.pkl", "rb"))

st.title("Laptop Price Predictor")

# Now we will take user input one by one as per our dataframe

# Brand
company = st.selectbox('Brand', data['Company'].unique())

# Type of laptop
lap_type = st.selectbox("Type", data['TypeName'].unique())

# Ram
ram = st.selectbox("Ram(in GB)", [2,4,6,8,12,16,24,32,64])

# Weight
weight = st.number_input("Weight of the Laptop")

# Touch screen
touchscreen = st.selectbox("TouchScreen", ['No', 'Yes'])

# IPS
ips = st.selectbox("IPS", ['No', 'Yes'])

# Screen size
screen_size = st.number_input('Screen Size')

# Resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

# CPU
cpu = st.selectbox('CPU', data['Cpu_Brand'].unique())

# HDD
hdd = st.selectbox('HDD(in GB)', [0,128,256,512,1024,2048])

# SSD
ssd = st.selectbox('SSD(in GB)', [0,8,128,256,512,1024])

# Flash storage
flash = st.selectbox('Flash_Storage(in GB)', [0,8,128,256,512,1024])

# GPU
gpu = st.selectbox('GPU', data['Gpu_brand'].unique())

# OS
os = st.selectbox('OS', data['OS'].unique())

# Prediction
if st.button('Predict Price'):
    ppi = None
    if touchscreen == "Yes":
        touchscreen = 1
    else:
        touchscreen = 0
        
    if ips == "Yes":
        ips = 1
    else:
        ips = 0
    
    if screen_size != 0:  # Check if screen_size is not zero to avoid division by zero
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res**2)) ** 0.5 / screen_size
    else:
        ppi = 0  # Set ppi to 0 if screen_size is zero
        
    query = np.array([company, lap_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, flash, gpu, os])
    query = query.reshape(1, -1)
    prediction = str(int(np.exp(pipe.predict(query)[0])))
    st.title("The predicted price of this configuration is " + prediction)
