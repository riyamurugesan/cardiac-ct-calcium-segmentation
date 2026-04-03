from predictive_mask_creation import CalciumPredictor
import numpy as np
from pathlib import Path


MODEL_PATH = 'Output/best_model.pth'
OUTPUT_DIR = Path('predictions')
OUTPUT_DIR.mkdir(exist_ok=True)

# create predictor
predictor = CalciumPredictor(MODEL_PATH,device='cuda')

ct = np.load('training_data/0/ct_volume.npy')
calcium = predictor.predict(ct)

print(f"Input shape: {ct.shape}")
print(f"Output shape: {calcium.shape}")
print(f"Calcium voxels: {calcium.sum()}")
print(f"Values: {np.unique(calcium)}")