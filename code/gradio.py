'''
This script is for the purpose of running a gradio powered inference on your local machine
'''

import torch
import gradio as gr
from PIL import Image
from torchvision import transforms

# Determine the device to use (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model and move it to the device
model_path = './ckpts/end.pkl'
model = torch.load(model_path, map_location=device)
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Define the image preprocessing function
def preprocess_image(image):
    if image is None:
        raise ValueError("Invalid image input: None")

    transform = transforms.Compose([
        transforms.Resize((333, 333)),    # Resize to 333x333
        transforms.CenterCrop(299),       # Center crop to 299x299
        transforms.ToTensor(),
        # Normalize as per training (scale images to [-1, 1])
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    img = transform(image).unsqueeze(0)  # Add batch dimension
    img = img.to(device)                 # Move tensor to the device
    return img

# Define the inference function
def infer(image):
    # Preprocess the input image
    input_tensor = preprocess_image(image)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)

    # Extract the classification logits
    logits = output[1]  # output[1] contains the logits

    # Apply sigmoid to get the probability
    prediction = torch.sigmoid(logits).item()

    threshold = 0.5 
    label = "Fake" if prediction > threshold else "Real"

    return label

# Set up the Gradio interface
iface = gr.Interface(fn=infer, inputs=gr.Image(type="pil"), outputs="text")
iface.launch()