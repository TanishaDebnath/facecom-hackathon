

```markdown
# Facecom Hackathon – Gender Classification & Face Recognition 🔍

🚀 A robust deep learning pipeline for gender classification and face recognition under adverse visual conditions, using the FACECOM dataset.


## 🧠 Task A – Gender Classification

### ✅ Overview:
- Input: Frontal face images
- Output: Binary gender classification (Male/Female)

### 🔧 Steps:
1. Face detection using OpenCV Haar cascades
2. Preprocessing images to 224×224
3. Fine-tuned `ResNet18` model
4. Training and validation using Torch + Sklearn

### 📊 Sample Results:
```

Accuracy: 91.5%
F1 Score: 91.2%

```
          precision    recall  f1-score   support
       0       0.84      0.69      0.76        78
       1       0.93      0.97      0.95       325
```

```

---

## 🧠 Task B – Face Recognition (Matching)

### ✅ Overview:
- Input: Validation faces vs database faces
- Output: Whether it matches a known identity or not

### 🔧 Steps:
1. Extract features using pretrained `ResNet18` (fc removed)
2. Cosine similarity matching between val/train
3. Match if similarity ≥ threshold (0.75 or 0.90)
4. Save results and calculate evaluation metrics

### 📊 Sample Results:
```

Accuracy: 84%
Precision: 100%
Recall: 84%
F1 Score: 91%

```
          precision    recall  f1-score   support
```

Non-Match       0.00      0.00      0.00         0
Match       1.00      0.84      0.91       403

````

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
````

Your `requirements.txt` should include:

```
torch
torchvision
opencv-python
scikit-learn
pandas
numpy
tqdm
Pillow
```

---

## ▶️ Running the Code

### 🟢 Task A:

```bash
cd Task_A
python facecom_task_a.py
```

### 🟢 Task B:

```bash
cd Task_B
python facecom_task_b.py
```

---

## 🧾 Notes

* You can optionally load `*.pth` models to skip retraining.
* Works on both CPU and GPU.
* Matching threshold is tunable via script (default: `0.75` for label, `0.90` for eval).

---

## 🙌 Team

👩‍💻 **Tanisha Debnath**
B.Tech CSE (AI), Institute of Engineering & Management
GitHub: [TanishaDebnath](https://github.com/TanishaDebnath)

---

