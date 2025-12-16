
# 🩻 Chest X-Ray Image Classification (Streamlit App)

A **Streamlit-based web application** for classifying chest X-ray images into **NORMAL** or **PNEUMONIA** using a **CNN/VGG16-trained deep learning model**.

This repository focuses on **model inference, visualization, and deployment**, not training.

---

## 🚀 Demo
🎥 **Demo Video: https://drive.google.com/file/d/1QfDRFUkW0OzDpgqwVWNyd1pe-qHSlj16/view?usp=drive_link** 

The demo shows real-time image upload and prediction through a Streamlit UI.

---

## 🧠 Model Details
- Deep Learning model trained using **CNN with VGG16 (Transfer Learning)**
- Binary classification:
  - `NORMAL`
  - `PNEUMONIA`
- Saved as `model.h5`

⚠️ Training code is **not included** in this repository.

---

## 📂 Repository Structure
```
chest_xray
├── Imageclassification.py          # Streamlit application
├── custom_pre_trained_model_15.h5  # Trained deep learning 
├── model.h5                        # Trained deep learning model
README.md
```

---

## 🖥️ Streamlit Visualization
The application allows users to:
- Upload a chest X-ray image
- Automatically preprocess the image
- Perform real-time prediction
- Display the predicted class on the UI

This demonstrates a complete **ML deployment workflow**.

---

## ▶️ How to Run Locally
```bash
pip install streamlit tensorflow pillow numpy
streamlit run Imageclassification.py
```

---

## 🛠 Tech Stack
- Python  
- TensorFlow / Keras  
- Streamlit  
- NumPy  
- Pillow (PIL)  

---

## 📊 Output Classes
- **NORMAL**
- **PNEUMONIA**

Results are displayed directly in the Streamlit interface.

---

## ⚠️ Disclaimer
This project is for **educational and demonstration purposes only**.  
It is **NOT** intended for real-world medical diagnosis or clinical use.

---

## 🎯 What This Project Demonstrates
- Deep learning model inference
- Transfer learning application
- Streamlit-based deployment
- End-to-end ML visualization

---

## 🔮 Future Improvements
- Grad-CAM visual explanations
- Confidence score display
- Multi-class classification
- Cloud deployment

---

## 👤 Author
**Ravi**  
AI & Data Science

---

## 📜 License
Educational use only.
