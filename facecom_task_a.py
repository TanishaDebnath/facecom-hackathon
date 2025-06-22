import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Create directory if not exists
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Crop and save faces using OpenCV Haar cascades
def crop_and_save_faces(source_dir, dest_dir):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count, failed = 0, 0
    for label in os.listdir(source_dir):
        source_path = os.path.join(source_dir, label)
        dest_path = os.path.join(dest_dir, label)
        ensure_dir(dest_path)

        for img_name in tqdm(os.listdir(source_path), desc=f"Processing {label}"):
            try:
                img_path = os.path.join(source_path, img_name)
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    cropped_face = img[y:y+h, x:x+w]
                    resized_face = cv2.resize(cropped_face, (224, 224))
                    save_path = os.path.join(dest_path, img_name)
                    Image.fromarray(resized_face).save(save_path)
                    count += 1
                else:
                    print(f"[!] No face detected in: {img_path}")
                    failed += 1
            except Exception as e:
                print(f"[!] Error processing {img_name}: {e}")
                failed += 1
    print(f"\n✅ Face cropping done. Total Processed: {count}, Failed: {failed}")

# Gender classification model (binary)
class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)

# Train and evaluate model
def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    model.to(device)
    for epoch in range(5):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            true_labels.extend(labels.numpy())

    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='weighted')
    print(f"\n✅ Gender Classification Report:")
    print(classification_report(true_labels, preds))
    print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}\n")

# ----------- Pipeline Start -------------
print("[Step 1] Preprocessing Faces for Gender Classification")
#dataset_root = "./Comys_Hackathon5"
dataset_root = "C:/Users/Tanisha Debnath/facecom_project/facecom_dataset/Comys_Hackathon5"

ensure_dir("./processed_gender/train")
ensure_dir("./processed_gender/val")

crop_and_save_faces(os.path.join(dataset_root, "Task_A", "train"), "./processed_gender/train")
crop_and_save_faces(os.path.join(dataset_root, "Task_A", "val"), "./processed_gender/val")

print("\n[Step 2] Training Gender Classification Model")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

gender_train_dataset = ImageFolder("./processed_gender/train", transform=transform)
gender_val_dataset = ImageFolder("./processed_gender/val", transform=transform)

gender_train_loader = DataLoader(gender_train_dataset, batch_size=32, shuffle=True)
gender_val_loader = DataLoader(gender_val_dataset, batch_size=32, shuffle=False)

model_gender = GenderClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_gender.parameters(), lr=0.001)

train_model(model_gender, gender_train_loader, gender_val_loader, criterion, optimizer, device)
print("\n🎉 Task A (Gender Classification) Completed!")


from sklearn.metrics import classification_report

# Optional: re-evaluate and show report
all_labels = []
all_preds = []

model_gender.eval()
with torch.no_grad():
    for images, labels in gender_val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model_gender(images)
        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

print("\n📊 Final Evaluation Report:")
print(classification_report(all_labels, all_preds, target_names=["Male", "Female"]))

# Save the model
torch.save(model_gender.state_dict(), "gender_model.pth")
print("✅ Model saved as gender_model.pth")
