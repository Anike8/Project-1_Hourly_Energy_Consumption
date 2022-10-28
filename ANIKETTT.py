

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
from matplotlib import pyplot as plt

st.title("Welcome to Energy Consumption Project")


st.sidebar.header('User Input Parameters')

no_days = st.sidebar.number_input("Insert Number of days", min_value=1, max_value=365, step=1)

with open("regresser.pkl", mode="rb") as f:
    model = pickle.load(f)

data=pd.read_csv('Daily_data.csv',index_col='Datetime',parse_dates=True)
data.rename({'PJMW_MW':'MW'},inplace=True,axis=1)
forecast_check_data = np.array(data['MW'][:'2018-07-04'][-7:])
z=forecast_check_data

for i in range(0,no_days):
    ck=z[-7:]
    ck=np.array([ck])
    lin_f_chk=model.predict(ck)
    z=np.append(z,lin_f_chk)
    i=+1
future_pred=z[-no_days:]

tab1, tab2 = st.tabs(["📈 Predicted Data","⛅ Graph"])

with tab1:
    Predict = pd.date_range(start='2018/08/04',periods=no_days,tz=None,freq = 'D')
    future_df = pd.DataFrame(index=Predict)
    future_df['Forecast'] = future_pred.tolist()
    st.write(future_df)
    
with tab2:
    fig, ax = plt.subplots()
    plt.figure(figsize=(14,5))
    ax.plot(future_df.index,future_df.values, label='Forecast', color="orange")
    ax.tick_params(axis='x', labelrotation = 100)
    plt.legend(fontsize=12, fancybox=True, shadow=True, frameon=True)
    plt.ylabel('Power consumption', fontsize=15)
    plt.xlabel('Date', fontsize=15)
    st.pyplot(fig)
