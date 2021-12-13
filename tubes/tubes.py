#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import base64
import numpy as np
import joblib
import plotly.figure_factory as ff



st.title('Analisis Hipertensi Menggunakan Algoritma Naïve Bayes') #Specify title of your app
st.write('data ini berasal dari Dinas Kesehatan Kabupaten Purwakarta .')
st.sidebar.markdown('## Data Import') #Streamlit also accepts markdown
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv") #data uploader

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.markdown('### Data Sample')
    st.write(data.head())
    id_col = st.sidebar.selectbox('Pick your ID column', options=data.columns)
    cat_features = st.sidebar.multiselect('Pick your categorical features', options=[c for c in data.columns], default = [v for v in data.select_dtypes(exclude=[int, float]).columns.values if v != id_col])
    clusters = data['cluster']
    df_p = data.drop(id_col, axis=1)
    if cat_features:
        df_p = pd.get_dummies(df_p, columns=cat_features) #OHE the categorical features
    prof = st.checkbox('Check to profile the clusters')

else:
    st.markdown("""
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <h1 style="color:#26608e;"> Upload your CSV file to begin clustering </h1>
    <h3 style="color:#f68b28;"> Customer segmentation </h3>
    """, unsafe_allow_html=True) 

    st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f'<h3 "> Features Overview </h3>',
            unsafe_allow_html=True)

#Memvisualisasikan data
#Menampilkan visualisasi bar plot dan pie chart untuk menghitung frekuensi dari kolom Di Diagnosis hipertensi dalam dataset data
f,ax=plt.subplots(1,2,figsize=(18,8))
df['DI DIAGNOSIS HIPERTENSI'].value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_xlabel('DI DIAGNOSIS HIPERTENSI')
ax[0].set_ylabel('JUMLAH')
sns.countplot('DI DIAGNOSIS HIPERTENSI',data=df,ax=ax[1])
ax[1].set_title('DI DIAGNOSIS HIPERTENSI')





