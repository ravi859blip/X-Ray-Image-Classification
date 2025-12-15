import streamlit as st
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image

# UI header
html_temp = """ 
    <div style="background-color:purple; padding:10px">
    <h2 style="color:white; text-align:center;"> X-Ray Image Classifier</h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

img_size = 100
CATEGORIES = ["NORMAL", "PNEUMONIA"]

# Load model (update path if needed)
MODEL_PATH = r"C:/Users/ASUS/X-Ray Image Classification/chest_xray/custom_pre_trained_model_15.h5"
model = tf.keras.models.load_model(MODEL_PATH)
st.write("✅ Model loaded")

def load_classifier():
    st.subheader("Upload an X-ray image to detect if it is Normal or Pneumonia")
    # allow common image types
    file = st.file_uploader(label="Choose an image file", type=['jpg', 'jpeg', 'png'])

    if file is not None:
        # Read image bytes and convert to RGB to guarantee 3 channels
        img = Image.open(BytesIO(file.read())).convert("RGB")
        img = img.resize((img_size, img_size))

        # Display the image (use the PIL image object)
        st.image(img, caption="Uploaded image", use_column_width=True)
        st.write("")

        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32)  # shape (100,100,3)
        img_array = img_array / 255.0
        input_batch = np.expand_dims(img_array, axis=0)  # shape (1,100,100,3)

        if st.button("Predict"):
            prediction = model.predict(input_batch)
            # Handle binary (sigmoid) vs categorical (softmax) outputs
            if prediction.shape[-1] == 1:
                # Sigmoid single-output (probability of class 1)
                prob_class1 = float(prediction[0][0])
                class_idx = 1 if prob_class1 >= 0.5 else 0
                confidence = prob_class1 if class_idx == 1 else 1 - prob_class1
            else:
                # Softmax multi-class
                class_idx = int(np.argmax(prediction[0]))
                confidence = float(prediction[0][class_idx])

            label = CATEGORIES[class_idx]
            st.write(f"**Prediction:** {label}")
            st.write(f"**Confidence:** {confidence*100:.2f}%")
            # Also print to console for debugging
            print("Raw model output:", prediction)

def main():
    load_classifier()

if __name__ == "__main__":
    main()