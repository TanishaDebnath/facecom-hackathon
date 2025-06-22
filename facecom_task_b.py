import os
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn.functional import normalize
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Identity()
model = model.to(device)
model.eval()

# Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Change to absolute path
root_path = "C:/Users/Tanisha Debnath/facecom_project/facecom_dataset"
train_dir = os.path.join(root_path, "processed_faces", "train")
val_dir = os.path.join(root_path, "processed_faces", "val")

def extract_embedding(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(img)
        return normalize(embedding, p=2, dim=1).cpu().numpy()[0]
    except:
        print(f"[!] Error reading image: {image_path}")
        return None

# Step 1: Build database
print("Building database from training set...")
db_embeddings = []
db_image_names = []

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"❌ Training folder not found: {train_dir}")

for person_folder in tqdm(os.listdir(train_dir)):
    person_path = os.path.join(train_dir, person_folder)
    if not os.path.isdir(person_path):
        continue
    for img_file in os.listdir(person_path):
        img_path = os.path.join(person_path, img_file)
        embedding = extract_embedding(img_path)
        if embedding is not None:
            db_embeddings.append(embedding)
            db_image_names.append(f"{person_folder}/{img_file}")

if len(db_embeddings) == 0:
    raise ValueError("⚠️ No embeddings found in training set!")

# Step 2: Match validation
print("Matching validation faces...")
results = []

if not os.path.exists(val_dir):
    raise FileNotFoundError(f"❌ Validation folder not found: {val_dir}")

for val_person_folder in tqdm(os.listdir(val_dir)):
    val_person_path = os.path.join(val_dir, val_person_folder)
    if not os.path.isdir(val_person_path):
        continue
    for val_img_file in os.listdir(val_person_path):
        val_img_path = os.path.join(val_person_path, val_img_file)
        val_embedding = extract_embedding(val_img_path)
        if val_embedding is None:
            continue
        
        similarities = [np.dot(val_embedding, db_emb) for db_emb in db_embeddings]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        best_match_img = db_image_names[best_idx]

        label = 1 if best_score >= 0.75 else 0

        results.append({
            "val_image": f"{val_person_folder}/{val_img_file}",
            "matched_train_image": best_match_img,
            "similarity": best_score,
            "label": label
        })

# Save results
results_df = pd.DataFrame(results)
output_path = os.path.join(root_path, "task_b_results.csv")
results_df.to_csv(output_path, index=False)
print(f"\n📄 Results saved to {output_path}")

# Evaluation
true_labels = results_df['label'].values
pred_labels = [1 if sim >= 0.90 else 0 for sim in results_df['similarity'].values]

print("\n📊 Evaluation Report:")
print(classification_report(true_labels, pred_labels, target_names=["Non-Match", "Match"]))
print(f"✅ Accuracy:  {accuracy_score(true_labels, pred_labels):.2f}")
print(f"🎯 Precision: {precision_score(true_labels, pred_labels):.2f}")
print(f"📈 Recall:    {recall_score(true_labels, pred_labels):.2f}")
print(f"⚖️ F1 Score:  {f1_score(true_labels, pred_labels):.2f}")

# Save model
torch.save(model.state_dict(), "face_recognition_model.pth")
print("✅ Model saved as face_recognition_model.pth")