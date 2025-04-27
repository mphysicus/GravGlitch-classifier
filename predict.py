import torch
import os
from PIL import Image
from model import GravitySpyResNet
from dataset import gravityspy_transform  # Shared transformations
from config import val_dir

def classify_glitches(list_of_img_paths):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load trained model
    model = GravitySpyResNet()
    checkpoint_path = '_checkpoints/best_weights.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Get class names from data directory
    data_root = val_dir
    try:
        class_names = sorted([entry.name for entry in os.scandir(data_root) 
                            if entry.is_dir()])
    except FileNotFoundError:
        raise RuntimeError(f"Data directory '{data_root}' not found")
    
    # Process images using shared transformations
    processed_images = []
    for img_path in list_of_img_paths:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image {img_path} does not exist")
            
        image = Image.open(img_path).convert('RGB')
        transformed = gravityspy_transform(image)
        processed_images.append(transformed)
    
    # Create batch tensor
    batch = torch.stack(processed_images).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(batch)
        _, predictions = torch.max(outputs, dim=1)
    
    return [class_names[idx] for idx in predictions.cpu()]