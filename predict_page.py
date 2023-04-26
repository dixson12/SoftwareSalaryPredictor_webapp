import streamlit as st
import pickle
import numpy as np
def load_model():
    with open('saved_steps.pkl','rb') as file:
        data=pickle.load(file)
    return data
data=load_model()
regressor=data['model']
le_country=data['le_country']
le_education=data['le_education']    
def show_prediction():
    st.title('Software Developer Salary prediction ')
    st.write("""We need some information to predict the salary""")

    Country=('United States of America','India','Germany','Canada','United Kingdom of Great Britain and Northern Ireland','Sweden','Poland','Italy','Australia','Netherlands','Spain','Brazil','France')
    Education=('Bachelors degree','Masters degree','Post Graduation','Less than bachelors')

    country=st.selectbox("Country",Country)
    education=st.selectbox("Education",Education)
    experience=st.slider("Years of Experience",0,50,3)
    predict =st.button('Predict Salary')
    if predict:
        X_input=np.array([[country,education,experience]])
        X_input[:,0]=le_country.transform(X_input[:,0])
        X_input[:,1]=le_education.transform(X_input[:,1])
        X_input=X_input.astype(float)
        y_pred=regressor.predict(X_input)
        st.subheader(f"The Estimated Annual Salary is ${y_pred[0]:0.2f}")

        

        