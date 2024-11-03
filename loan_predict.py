import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import time
from PIL import Image

st.title("Loan Approval Predictor")

df=pd.read_csv("loan_approval_dataset.csv")
image=Image.open("loan-approved.jpeg")
st.image(image,caption="Loan",use_column_width=True)

dependents=st.sidebar.slider("Enter the number of dependents",1,10,3)
income=st.sidebar.slider("Enter the income annum",0,int(df[" income_annum"].max()),1000)
amount=st.sidebar.slider("Enter the loan ammount",0,int(df[" loan_amount"].max()),3)
term=st.sidebar.slider("Enter the loan term",0,int(df[" loan_term"].max()),3)
score=st.sidebar.slider("Enter the cibil score",0,int(df[" cibil_score"].max()),5)
residential=st.sidebar.slider("Enter the residential assets value",0,int(df[" residential_assets_value"].max()),20)
commercial=st.sidebar.slider("Enter the commercial assets value",0,int(df[" commercial_assets_value"].max()),20)
luxury=st.sidebar.slider("Enter the luxury assets value",0,int(df[" luxury_assets_value"].max()),20)
bank=st.sidebar.slider("Enter the bank asset value",0,int(df[" bank_asset_value"].max()),4)

x=df.drop(columns=[" loan_amount"," education"," self_employed"," loan_status"])
y=df[" loan_amount"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

regressor=KNeighborsRegressor(n_neighbors=dependents)
regressor.fit(x_train_scaled,y_train)

ui_df=pd.DataFrame({'Dependents':[dependents],
                    'Income Annum':[income],
                    'Loan Amount':[amount],
                    'Loan Term':[term],
                    'Cibil Score':[score],
                    'Residential Assets Value':[residential],
                    'Commerical Assets Value':[commercial],
                    'Luxury Assets Value':[luxury],
                    'Bank Asset Value':[bank]},columns=x.columns)

ui_scaled=scaler.transform(ui_df)

with st.spinner('Calculating...'):
    time.sleep(2)
    predicted_loan_ammount=regressor.predict(ui_scaled)

st.write(f'The predicted loan ammount for the given input is:{predicted_loan_ammount[0]}')