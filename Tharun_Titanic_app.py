#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import streamlit as st
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Load trained model
log_model = pickle.load(open('log19.pkl', 'rb'))


# In[3]:


st.title("Logistic Regression Deployment")


# In[10]:


def user_input_perameter():
    pclass=st.number_input("Enter Passenger Class:",1,3)
    sex = st.selectbox("Enter Sex:Male=1, Female=0", [0,1])
    age = st.slider("Enter Age", 0, 100, 30)
    sibsp=st.number_input("Siblings/Spouses",0,10)
    parch=st.number_input("Parents/children",0,8)
    fare = st.number_input("Fare", 0.0, 600.0, 10.0)
    embarked=st.selectbox("Embarked",[1,2])
    
    dict1={'Pclass':pclass,'Sex':sex,'Age':age,'SibSp':sibsp,'Parch':parch,'Fare':fare,'Embarked':embarked}
    features=pd.DataFrame(dict1,index=[0])
    return features

df=user_input_perameter()

# Predict
pred= log_model.predict(df)
pred_prob=log_model.predict_proba(df)
button=st.button('Predict')
if button is True:
    st.subheader("Predicted")
    st.write('Survived' if pred_prob[0][1]>=0.5 else 'Not Survived')
    st.write(pred)
    st.write(pred_prob)


# In[ ]:





# # **Interview Questions**

# # 1. What is the difference between precision and recall?

# In[ ]:





# # 2. What is cross-validation, and why is it important in binary classification?

# In[ ]:




