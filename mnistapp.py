import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Define the class names for Fashion MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Load your trained Keras model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fashionmnist_model.h5")

model = load_model()

# App title
st.title("👕 Fashion MNIST Classifier")
st.write("Upload a **28x28 grayscale image** to predict the clothing category.")

# Upload image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded Image (resized to 28x28)", width=150)

    # Resize and preprocess
    image = image.resize((28, 28))  # Ensure 28x28
    image_array = np.array(image) / 255.0  # Normalize
    image_array = image_array.reshape(1, 28, 28)  # Add batch dimension

    # Make prediction
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    # Show prediction
    st.markdown(f"### 🧠 Predicted: `{class_names[predicted_class]}`")
    st.markdown(f"**Confidence:** {confidence * 100:.2f}%")

    # Show full probability bar chart
    st.subheader("Prediction Confidence")
    st.bar_chart(predictions[0])
