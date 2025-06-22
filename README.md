

```markdown
# Facecom Hackathon – Gender Classification & Face Recognition 🔍

🚀 A robust deep learning pipeline for gender classification and face recognition under adverse visual conditions, using the FACECOM dataset.

---

## 🧠 Task A – Gender Classification

### ✅ Overview:
- Input: Frontal face images
- Output: Binary gender classification (Male/Female)

### 🔧 Steps:
1. Face detection using OpenCV Haar cascades
2. Preprocessing images to 224×224
3. Fine-tuned `ResNet18` model
4. Training and validation using PyTorch + Sklearn

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
- Input: Validation faces vs training database faces
- Output: Whether it matches a known identity or not

### 🔧 Steps:
1. Extract embeddings using pretrained `ResNet18` (fc removed)
2. Use cosine similarity to match validation vs training faces
3. Match if similarity ≥ threshold (default 0.75 for labeling)
4. Save results and calculate evaluation metrics

### 📊 Sample Results:

```

Accuracy: 84%
Precision: 100%
Recall: 84%
F1 Score: 91%

```
          precision    recall  f1-score   support


Non-Match       0.00      0.00      0.00         0
Match       1.00      0.84      0.91       403

````

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
````

### `requirements.txt` includes:

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

## 📥 Download Pretrained Models

Download and save the following `.pth` model files in the respective folders:

* 🧑‍🦰 [gender\_model.pth – Task A](https://drive.google.com/file/d/1ChcBiq-dpOjkJcRyXu18S-uMxC8LpkXR/view?usp=sharing) → Place in `Task_A/`
* 🧑‍🦱 [face\_recognition\_model.pth – Task B](https://drive.google.com/file/d/1Xkwl3xrfl2MUC5zJWC9XZRfP3pOuwvBR/view?usp=sharing) → Place in `Task_B/`

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

* You can optionally load `.pth` models to skip retraining.
* Code is GPU-compatible (runs on CPU as fallback).
* Thresholds are tunable within scripts (`0.75` for labeling, `0.90` for evaluation).

---

## 🙌 Author

👩‍💻 **Tanisha Debnath**
B.Tech CSE (AI), Institute of Engineering & Management
🔗 GitHub: [TanishaDebnath](https://github.com/TanishaDebnath)

---

```

