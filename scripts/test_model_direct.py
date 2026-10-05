import torch
import numpy as np
from pathlib import Path
from utils.dl_engine import ExerciseAIEngine # لما نكريته

# سكريبت بسيط جداً بياخد ملف .npz عشوائي ويجرب الموديل عليه
def test_model():
    # 1. اختار تمرين (مثلاً squat)
    exercise = "squat"
    checkpoint = f"checkpoints/dl_model_{exercise}.pt"
    
    # 2. حمل الموديل
    engine = ExerciseAIEngine(checkpoint)
    
    # 3. هات أي ملف .npz من الـ processed
    # غير المسار ده للمسار الفعلي عندك
    data_path = Path("data/processed/squat/squat_10.npz") 
    data = np.load(data_path)
    keypoints = data["keypoints"] # (N, 17, 2)
    
    print(f"Testing model on {len(keypoints)} frames...")
    
    # 4. جرب أول 10 فريمات
    for i in range(10):
        reps, state, conf = engine.update(keypoints[i])
        print(f"Frame {i} -> State: {state}, Conf: {conf:.2f}")

if __name__ == "__main__":
    test_model()