import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warning
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import time
# Configure page with modern theme
st.set_page_config(
    page_title="AI Pet Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# Custom CSS for modern UI
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# Create custom CSS
def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }
        
        .main-header {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .sub-header {
            font-size: 1.2rem;
            font-weight: 400;
            color: #6c757d;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .card {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            margin-bottom: 25px;
            border: 1px solid #f0f2f5;
        }
        
        .result-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
            margin-bottom: 25px;
            text-align: center;
        }
        
        .prediction-text {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 15px 0;
        }
        
        .confidence-text {
            font-size: 1.5rem;
            font-weight: 500;
            color: #495057;
            margin: 10px 0;
        }
        
        .upload-area {
            border: 2px dashed #dee2e6;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            background-color: #f8f9fa;
            transition: all 0.3s ease;
        }
        
        .upload-area:hover {
            border-color: #4ECDC4;
            background-color: #f0f9fa;
        }
        
        .stButton button {
            background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .footer {
            text-align: center;
            margin-top: 30px;
            color: #6c757d;
            font-size: 0.9rem;
        }
        
        .loading-spinner {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100px;
        }
        
        .loading-spinner::after {
            content: "";
            width: 40px;
            height: 40px;
            border: 4px solid rgba(0, 0, 0, 0.1);
            border-radius: 50%;
            border-top-color: #4ECDC4;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .progress-container {
            margin: 20px 0;
        }
        
        .stProgress > div > div > div > div {
            background-color: #4ECDC4;
        }
        
        .details-card {
            background-color: #e6f7ff;
            border-radius: 10px;
            padding: 15px;
            margin-top: 15px;
            border-left: 4px solid #4ECDC4;
            color: #333;
        }
        
        .details-card p {
            color: #333;
            margin: 8px 0;
        }
        
        .details-card b {
            color: #000;
        }
        
        .reset-button {
            margin-top: 20px;
            width: 100%;
        }
        
        .streamlit-expanderHeader {
            background-color: #f0f9fa;
            border-radius: 10px;
            padding: 10px;
        }
        
        .warning-message {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 15px 0;
            border-radius: 5px;
            color: #856404;
        }
    </style>
    """, unsafe_allow_html=True)
# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('custom_classifier.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
# Preprocess the image
def preprocess_image(image):
    # Convert to RGB if it's RGBA (4 channels)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Resize image to match model input size
    image = image.resize((160, 160))
    
    # Convert to array and preprocess
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
# Main app
def main():
    inject_custom_css()
    
    # Header section
    st.markdown('<div class="main-header">AI Pet Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload an image to identify if it\'s a cat or a dog</div>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Upload Image")
        
        # File uploader with custom styling
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"],
            key="file_uploader",
            help="Upload a clear image of a cat or dog"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Add a button to trigger prediction
            predict_button = st.button("Classify Image", key="predict")
            
            if predict_button:
                # Show loading spinner
                with st.spinner('Analyzing image...'):
                    # Add a custom loading spinner
                    st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)
                    
                    try:
                        # Preprocess and predict
                        processed_image = preprocess_image(image)
                        raw_prediction = model.predict(processed_image)[0][0]
                        
                        # Apply sigmoid to get probability if needed
                        if raw_prediction < 0 or raw_prediction > 1:
                            prediction = tf.nn.sigmoid(raw_prediction).numpy()
                        else:
                            prediction = raw_prediction
                        
                        # Determine result with adjusted threshold
                        if prediction > 0.7:  # Increased threshold for dog
                            result = "Dog"
                            confidence = float(prediction)
                            emoji = "🐶"
                            color = "#FF6B6B"
                        elif prediction < 0.3:  # Decreased threshold for cat
                            result = "Cat"
                            confidence = float(1 - prediction)
                            emoji = "🐱"
                            color = "#4ECDC4"
                        else:
                            result = "Uncertain"
                            confidence = float(abs(prediction - 0.5) * 2)  # Confidence based on distance from 0.5
                            emoji = "❓"
                            color = "#FFC107"
                        
                        # Convert confidence to percentage
                        confidence_percent = confidence * 100
                        
                        # Store results in session state
                        st.session_state.prediction_result = {
                            "result": result,
                            "confidence": confidence,
                            "confidence_percent": confidence_percent,
                            "emoji": emoji,
                            "color": color,
                            "raw_score": raw_prediction,
                            "final_prediction": prediction
                        }
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        st.session_state.prediction_error = str(e)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Prediction Result")
        
        # Display error if prediction failed
        if "prediction_error" in st.session_state:
            st.error(f"Prediction failed: {st.session_state.prediction_error}")
            
        # Display prediction results if available
        elif "prediction_result" in st.session_state:
            result = st.session_state.prediction_result
            
            # Result card with gradient background
            st.markdown(f"""
            <div class="result-card">
                <div class="prediction-text" style="color: {result['color']};">
                    {result['emoji']} {result['result']}
                </div>
                <div class="confidence-text">
                    Confidence: {result['confidence_percent']:.2f}%
                </div>
                <div class="progress-container">
                    <div style="background-color: #e9ecef; border-radius: 10px; height: 20px; overflow: hidden;">
                        <div style="background-color: {result['color']}; width: {result['confidence_percent']}%; height: 100%; border-radius: 10px; transition: width 1s ease-in-out;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show warning if result is uncertain
            if result['result'] == "Uncertain":
                st.markdown("""
                <div class="warning-message">
                    <b>Note:</b> The model is uncertain about this prediction. The image may not clearly show a cat or dog, or the model may not have been trained on similar images.
                </div>
                """, unsafe_allow_html=True)
            
            # Details section
            with st.expander("See details"):
                st.markdown(f"""
                <div class="details-card">
                    <p><b>Raw model output:</b> {result['raw_score']:.4f}</p>
                    <p><b>Final probability:</b> {result['final_prediction']:.4f}</p>
                    <p><b>Model:</b> MobileNetV2 with transfer learning</p>
                    <p><b>Input size:</b> 160x160 pixels</p>
                    <p><b>Thresholds:</b> Dog > 0.7, Cat < 0.3, Uncertain = 0.3-0.7</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Add a reset button with custom styling
            st.markdown('<div class="reset-button">', unsafe_allow_html=True)
            if st.button("Classify Another Image", key="reset_button"):
                if "prediction_result" in st.session_state:
                    del st.session_state.prediction_result
                if "prediction_error" in st.session_state:
                    del st.session_state.prediction_error
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Placeholder when no prediction is available
            st.markdown("""
            <div style="text-align: center; padding: 40px; color: #6c757d;">
                <p>Upload an image and click "Classify Image" to see the prediction result here.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="footer">Developed by ASAD-AZIZ | AI Pet Classifier © 2025</div>', unsafe_allow_html=True)
if __name__ == "__main__":
    main()