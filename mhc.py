# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 23:20:11 2023

@author: rukmi
"""
import sklearn
import pickle
import streamlit as st
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() # instantiate a MinMaxScaler object
#from streamlit_option_menu import option_menu
#with open('file.pkl', 'rb') as file:
      
    # Call load method to deserialze
    #myvar = pickle.load(file)
#loading
loaded_model=pickle.load(open('C:/Users/rukmi/OneDrive/Desktop/project/IPD2/MHC/mhc.sav','rb'))
#page title
st.title('Mental Health Companion')

# =============================================================================
# Age=st.text_input("Enter your Age")
# Gender=st.text_input("Gender")
# family_history=st.text_input("Do you have a family history of mental illness?")
# benefits=st.text_input("Benefits")
# care_options=st.text_input("care Options")
# anonymity=st.text_input("Anonymity")
# leave=st.text_input("Leave")
# work_interfere=st.text_input("Work interference")
# =============================================================================

#values of selectboxes
genders=['female', 'male', 'other ']
family=['No', 'Yes']
benefits1=["Don't know", 'No', 'Yes']
careop=['No', 'Not sure', 'Yes']
anynom=["Don't know", 'No', 'Yes']
leaves=["Don't know", 'Somewhat difficult', 'Somewhat easy', 'Very difficult', 'Very easy']
work=["Don't know", 'Never', 'Often', 'Rarely', 'Sometimes']

#taking input from users
Age=st.number_input("Enter your Age")
Gender=st.selectbox("Select your Gender",genders)
family_history=st.selectbox("Do you have a family history of mental illness?",family)
benefits=st.selectbox("Does your employer provide mental health benefits?",benefits1)
care_options=st.selectbox("Do you know the options for mental health care?",careop)
anonymity=st.selectbox("Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?",anynom)
leave=st.selectbox("How easy is it for you to take medical leave for a mental health condition?",leaves)
work_interfere=st.selectbox("If you have a mental health condition, do you feel that it interferes with your work?",work)




#Mapping values to variable options
a = scaler.fit_transform([[Age]])
if Gender in genders:
    g = list(genders).index(Gender)
#family_history
if family_history in family:
    f = list(family).index(family_history)
#benefits
if benefits in benefits1:
    b=list(benefits1).index(benefits)
#care_options
if care_options in careop:
    c=list(careop).index(care_options)
#anynomity
if anonymity in anynom:
    a=list(anynom).index(anonymity)
#levae
if leave in leaves:
    l=list(leaves).index(leave)
#Work_interefere
if work_interfere in work:
    w=list(work).index(work_interfere)
    

#code for Prediction
mhc_pred=''
#button for prediction
if st.button("Check"):
    #mhc_pred=loaded_model.predict([[Age, Gender, family_history, benefits, care_options, anonymity, leave, work_interfere]])
    mhc_pred=loaded_model.predict([[a, g, f, b, c, a, l, w]])
    
    if(mhc_pred[0]==1):
        mhc_pred="Companion says, This Person needs mental health treatment."
    else:
        mhc_pred="Companion says, This Person doesn't require mental health treatment."
st.success(mhc_pred)
