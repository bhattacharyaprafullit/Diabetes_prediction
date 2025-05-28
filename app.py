import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
# Add this at the top of your app.py file
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('trained_model.sav', 'rb'))
        return model
    except FileNotFoundError:
        st.error("Model file 'trained_model.sav' not found. Please ensure the file is in the same directory.")
        return None

# Create scaler with pre-calculated statistics from the diabetes dataset
@st.cache_resource
def create_scaler():
    """
    Creates a StandardScaler with pre-calculated mean and scale values
    from the Pima Indians Diabetes dataset
    """
    scaler = StandardScaler()
    
    # These are the actual mean and standard deviation values from the diabetes dataset
    # Calculated from the original Pima Indians Diabetes dataset
    mean_values = np.array([3.8459677, 120.89451, 69.10553, 20.536398, 
                           79.79956, 31.992578, 0.4718763, 33.240885])
    
    scale_values = np.array([3.3695781, 31.97261, 19.355807, 15.952218,
                            115.24401, 7.8841603, 0.33132287, 11.760232])
    
    # Set the scaler parameters manually
    scaler.mean_ = mean_values
    scaler.scale_ = scale_values
    scaler.var_ = scale_values ** 2
    scaler.n_features_in_ = 8
    scaler.feature_names_in_ = np.array(['Pregnancies', 'Glucose', 'BloodPressure', 
                                        'SkinThickness', 'Insulin', 'BMI', 
                                        'DiabetesPedigreeFunction', 'Age'])
    
    return scaler

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Diabetes Prediction App",
        page_icon="🩺",
        layout="wide"
    )
    
    # App title and description
    st.title("🩺 Diabetes Prediction System")
    st.markdown("---")
    st.write("This app predicts whether a person has diabetes based on various health parameters.")
    
    # Load model and create scaler
    model = load_model()
    scaler = create_scaler()
    
    if model is None:
        st.stop()
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Information")
        
        # Input fields
        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0,
            max_value=20,
            value=0,
            help="Number of times pregnant"
        )
        
        glucose = st.number_input(
            "Glucose Level",
            min_value=0.0,
            max_value=300.0,
            value=120.0,
            help="Plasma glucose concentration (mg/dL)"
        )
        
        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0.0,
            max_value=200.0,
            value=80.0,
            help="Diastolic blood pressure (mm Hg)"
        )
        
        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            help="Triceps skin fold thickness (mm)"
        )
    
    with col2:
        st.subheader("Additional Parameters")
        
        insulin = st.number_input(
            "Insulin Level",
            min_value=0.0,
            max_value=1000.0,
            value=80.0,
            help="2-Hour serum insulin (mu U/ml)"
        )
        
        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=70.0,
            value=25.0,
            help="Body mass index (weight in kg/(height in m)^2)"
        )
        
        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.001,
            format="%.3f",
            help="Diabetes pedigree function"
        )
        
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=25,
            help="Age in years"
        )
    
    # Center the prediction button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_button = st.button("🔍 Predict Diabetes", type="primary", use_container_width=True)
    
    # Make prediction
    if predict_button:
        # Create input array
        input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, 
                              insulin, bmi, diabetes_pedigree, age])
        
        # Reshape for prediction
        input_data_reshaped = input_data.reshape(1, -1)
        
        # Apply scaling
        input_data_scaled = scaler.transform(input_data_reshaped)
        
        # Make prediction
        try:
            prediction = model.predict(input_data_scaled)
            prediction_proba = model.predict_proba(input_data_scaled) if hasattr(model, 'predict_proba') else None
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            if prediction[0] == 0:
                st.success("✅ The person is predicted to be **NON-DIABETIC**")
                st.balloons()
            else:
                st.error("⚠️ The person is predicted to be **DIABETIC**")
            
            # Show probability if available
            if prediction_proba is not None:
                prob_non_diabetic = prediction_proba[0][0] * 100
                prob_diabetic = prediction_proba[0][1] * 100
                
                st.subheader("Prediction Confidence")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Non-Diabetic Probability", f"{prob_non_diabetic:.1f}%")
                
                with col2:
                    st.metric("Diabetic Probability", f"{prob_diabetic:.1f}%")
            
            # Show input summary
            with st.expander("📊 Input Summary"):
                input_df = pd.DataFrame({
                    'Parameter': ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
                                'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age'],
                    'Value': [pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, diabetes_pedigree, age]
                })
                st.dataframe(input_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
    
    # Add disclaimer
    st.markdown("---")
    st.markdown("""
    **⚠️ Disclaimer:** This prediction is based on machine learning and should not be used as a substitute for professional medical advice. 
    Please consult with a healthcare professional for proper medical diagnosis and treatment.
    """)
    
    # Add information about the model
    with st.expander("ℹ️ About this Model"):
        st.write("""
        This diabetes prediction model is trained using the Pima Indians Diabetes Database.
        
        **Features used for prediction:**
        - Number of pregnancies
        - Glucose level
        - Blood pressure
        - Skin thickness
        - Insulin level
        - BMI (Body Mass Index)
        - Diabetes pedigree function
        - Age
        
        **Model:** Support Vector Machine (SVM) with linear kernel
        """)

if __name__ == "__main__":
    main()