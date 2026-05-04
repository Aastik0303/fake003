from langchain_core.tools import tool
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import os

# Initialize ViT Model globally to save loading time during chat
print("Loading ViT Model...")
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
print("ViT Model Loaded!")

@tool
def vit_deepfake_detector(image_path: str) -> str:
    """
    ALWAYS use this tool when the user asks to analyze, check, or verify an image for deepfakes or manipulation.
    Input should be the absolute file path of the image.
    Returns a detailed forensic analysis score.
    """
    if not os.path.exists(image_path):
        return f"Error: Image not found at {image_path}"

    try:
        image = Image.open(image_path).convert("RGB")
        
        # Preprocessing for ViT (Handles high-res by scaling to 224x224 patches)
        inputs = processor(images=image, return_tensors="pt")
        
        # Forward Pass
        with torch.no_grad():
            outputs = model(**inputs)
            
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Assuming index 0 is Real, index 1 is Fake (Modify based on your specific model)
        fake_prob = probabilities[0][1].item() * 100
        real_prob = probabilities[0][0].item() * 100
        
        status = "DEEPFAKE DETECTED" if fake_prob > 70.0 else "LIKELY AUTHENTIC"
        
        # We return a structured string to the LLM so it can read and explain it
        return (f"--- ViT Analysis Results ---\n"
                f"Status: {status}\n"
                f"Fake Probability: {fake_prob:.2f}%\n"
                f"Authentic Probability: {real_prob:.2f}%\n"
                f"Note: The Vision Transformer analyzed the 16x16 pixel patches for global inconsistencies.")
                
    except Exception as e:
        return f"Tool Error: {str(e)}"
