import streamlit as st
import numpy as np
import pickle

style = """<div style='background-color:white; padding:12px'>
              <h1 style='color:black'>House Price Prediction App - ICBT</h1>
       </div>"""
st.markdown(style, unsafe_allow_html=True)



with open('catboost_enco.sav', 'rb') as f:
    model = pickle.load(f)
    
    
def predict_price(OverallQual, YearBuilt, YearRemodAdd, TotalBsmtSF, 
                  FirstFlrSF, GrLivArea, FullBath, TotRmsAbvGrd, 
                  GarageCars, GarageArea):
    inputs = [OverallQual, YearBuilt, YearRemodAdd, TotalBsmtSF, 
              FirstFlrSF, GrLivArea, FullBath, TotRmsAbvGrd, 
              GarageCars, GarageArea]
    inputs = np.array(inputs).reshape(1,-1)
    prediction = model.predict(inputs)[0]
    return prediction

def main():
    # Define the app title and a brief description
    #st.title("Advanced House Price Prediction")
    st.subheader("Azaam Abubacker | Cardiff Metropolitan University")
    st.text("Please enter the following information to get a price estimate")

    # Add input fields for user input
    OverallQual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    YearBuilt = st.slider("Year Built", 1900, 2022, 2000)
    YearRemodAdd = st.slider("Year Remodeled", 1900, 2022, 2010)
    TotalBsmtSF = st.number_input("Total Basement Area (in sqft)", value=1000, step=100)
    FirstFlrSF = st.number_input("First Floor Area (in sqft)", value=1000, step=100)
    GrLivArea = st.number_input("Living Area (in sqft)", value=1500, step=100)
    FullBath = st.slider("Number of Full Bathrooms", 0, 4, 2)
    TotRmsAbvGrd = st.slider("Total Rooms Above Grade", 2, 14, 6)
    GarageCars = st.slider("Number of Cars in Garage", 0, 4, 2)
    GarageArea = st.number_input("Garage Area (in sqft)", value=500, step=50)

    # Add a button to trigger the prediction
    if st.button("Predict Price"):
        price = predict_price(OverallQual, YearBuilt, YearRemodAdd, TotalBsmtSF, 
                              FirstFlrSF, GrLivArea, FullBath, TotRmsAbvGrd, 
                              GarageCars, GarageArea)
        st.success(f"The estimated price of the house is ${price:,.2f}")

if __name__ == "__main__":
    main()




