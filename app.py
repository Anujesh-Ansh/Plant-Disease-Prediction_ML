import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

# Load the trained model
model_path = '/Users/anujeshansh/Downloads/Plant_disease_prediction/trained_plant_disease_model.keras'
cnn = tf.keras.models.load_model(model_path)

# Define class names corresponding to your model's output
class_names = [ 
    'Apple Scab', 'Apple Black Rot', 'Cedar Apple Rust', 'Healthy Apple',
    'Healthy Blueberry', 'Cherry Powdery Mildew', 'Healthy Cherry', 'Corn Gray Leaf Spot',
    'Corn Common Rust', 'Corn Northern Leaf Blight', 'Healthy Corn', 'Grape Black Rot',
    'Potato Early Blight', 'Potato Late Blight', 'Tomato Bacterial Spot', 'Tomato Early Blight',
    'Tomato Late Blight', 'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot', 'Tomato Spider Mites',
    'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato Mosaic Virus', 'Healthy Tomato'
    # Add all other classes as per your model training...
]

# Define remedies for diseases
disease_remedies = {
    'Apple Scab': "Apply a fungicide like copper sulfate or lime sulfur.",
    'Apple Black Rot': "Remove infected branches and apply a fungicide.",
    'Cedar Apple Rust': "Use a fungicide such as myclobutanil and remove infected plant debris.",
    'Healthy Apple': "No treatment needed.",
    'Healthy Blueberry': "No treatment needed.",
    'Cherry Powdery Mildew': "Use a fungicide like sulfur or potassium bicarbonate.",
    'Healthy Cherry': "No treatment needed.",
    'Corn Gray Leaf Spot': "Apply fungicides and practice crop rotation.",
    'Corn Common Rust': "Use resistant corn varieties and apply fungicides.",
    'Corn Northern Leaf Blight': "Apply fungicides and practice crop rotation.",
    'Healthy Corn': "No treatment needed.",
    'Grape Black Rot': "Use fungicides and remove infected plant debris.",
    'Potato Early Blight': "Use fungicides and remove infected leaves.",
    'Potato Late Blight': "Apply fungicides and avoid overhead irrigation.",
    'Tomato Bacterial Spot': "Remove infected plants and apply copper-based fungicides.",
    'Tomato Early Blight': "Apply fungicides and practice crop rotation.",
    'Tomato Late Blight': "Use fungicides and remove infected plants.",
    'Tomato Leaf Mold': "Improve air circulation and apply fungicides.",
    'Tomato Septoria Leaf Spot': "Remove infected leaves and apply fungicides.",
    'Tomato Spider Mites': "Use miticides and introduce natural predators.",
    'Tomato Target Spot': "Use fungicides and remove infected leaves.",
    'Tomato Yellow Leaf Curl Virus': "Control whiteflies and use resistant varieties.",
    'Tomato Mosaic Virus': "Remove infected plants and disinfect tools.",
    'Healthy Tomato': "No treatment needed."
    # Add remedies for other classes as needed
}

# Streamlit UI
st.title('Plant Disease Prediction')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (128, 128))
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize and create batch axis

    # Make predictions
    predictions = cnn.predict(img_array)
    result_index = np.argmax(predictions[0])
    
    # Debugging: Print result_index and number of classes
    st.write(f"Result index: {result_index}")
    st.write(f"Number of classes in class_names list: {len(class_names)}")

    # Ensure result_index is within the bounds of class_names
    if result_index < len(class_names):
        model_prediction = class_names[result_index]
    else:
        model_prediction = "Unknown disease"
    
    # Get remedy
    remedy = disease_remedies.get(model_prediction, "No remedy found for this plant.")

    # Display results
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write(f"Predicted disease: {model_prediction}")
    st.write(f"Recommended remedy: {remedy}")
