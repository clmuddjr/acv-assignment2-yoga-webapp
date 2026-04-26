# Yoga Pose Classification Web Application

This Streamlit web application allows a user to upload a yoga image, choose among multiple quantized TFLite model versions, and view the predicted yoga pose.

## Included Models

- Assignment 1 Baseline CNN
- EfficientNetB0 Frozen Only
- ResNet50 Frozen Only
- MobileNetV2 Frozen Only

## Features

- Image upload
- Dropdown selector for model choice
- Quantized TFLite inference
- Predicted class display
- Top-3 prediction visualization

## Run Locally

From the `webapp` folder, run:

```bash
streamlit run app.py