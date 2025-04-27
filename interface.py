# Model
from model import GravitySpyResNet as TheModel

# Training function
from train import train_model as the_trainer

# Prediction function
from predict import classify_glitches as the_predictor

# Dataset and DataLoader
from dataset import GravitySpyDataset as TheDataset
from dataset import gravityspy_loader as the_dataloader