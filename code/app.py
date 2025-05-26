'''
This script is intended to host a streamlit app for a live demo of third-eye
'''

import torch
from PIL import Image
from torchvision import transforms
import streamlit as st

# Force CPU for Streamlit Cloud compatibility
device = torch.device("cpu")

@st.cache_resource
def load_model(path="./ckpts/end.pkl"):
    model = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.to(device).eval()
    return model

model = load_model()

def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((40,40)),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor

def predict(image: Image.Image):
    if image is None:
        return "No image uploaded."
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
    logits = output[1]
    prob = torch.sigmoid(logits).item()
    return "Fake" if prob > 0.5 else "Real"

st.title("Third-Eye : Frequency-Aware Deepfake Identification")
uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    if st.button("Classify"):
        label = predict(img)
        st.subheader(f"Prediction: {label}")
