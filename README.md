
````markdown
# 🧠 Comsys Hackathon 2025 – Robust Face & Gender Recognition

This repository contains my solution for the **Comsys Hackathon 2025** conducted by **Jadavpur University**, focused on robust face recognition and gender classification under adverse visual conditions.

## 📁 Tasks Overview

### 🔹 Task A – Gender Classification (Binary)
Build a CNN-based classifier using face images to classify gender (`Male`, `Female`).

- ✅ ResNet-18 fine-tuned
- ✅ Face preprocessing using OpenCV Haar Cascades
- ✅ Evaluation on processed validation set

### 🔹 Task B – Face Recognition (Multi-Class Matching)
Match distorted face images against a known database using deep features (embeddings).

- ✅ ResNet-18 embeddings
- ✅ Cosine similarity-based matching
- ✅ Multi-class verification using threshold-based labeling

---

## 🛠 Technologies Used

- Python 🐍
- PyTorch 🔥
- OpenCV
- scikit-learn
- torchvision
- tqdm, pandas, PIL

---

## 📊 Final Results

### 🚺 Task A: Gender Classification

| Metric        | Training (%) | Validation (%) |
|---------------|--------------|----------------|
| Accuracy      | 97.94        | 90.82          |
| Precision     | 98.10        | 91.00          |
| Recall        | 97.94        | 90.82          |
| F1 Score      | 97.98        | 90.84          |

---

### 🧑‍🤝‍🧑 Task B: Face Recognition

| Metric        | Training (%) | Validation (%) |
|---------------|--------------|----------------|
| Accuracy      | 100.00       | 84.00          |
| Precision     | 100.00       | 100.00         |
| Recall        | 100.00       | 84.00          |
| F1 Score      | 100.00       | 91.00          |

---

## 🚀 How to Run

1. **Clone this repository**
   ```bash
   git clone https://github.com/TanishaDebnath/facecom-hackathon.git
   cd facecom-hackathon
````

2. **Set up a virtual environment**

   ```bash
   python -m venv venv
   venv\Scripts\activate   # For Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run Gender Classification (Task A)**

   ```bash
   cd facecom_dataset/Task_A
   python facecom_task_a.py
   ```

5. **Run Face Recognition (Task B)**

   ```bash
   cd ../Task_B
   python facecom_task_b.py
   ```

### 6. 🔐 **Saved Models**
- [`gender_model.pth`](https://drive.google.com/file/d/1ChcBiq-dpOjkJcRyXu18S-uMxC8LpkXR/view?usp=sharing) – Trained binary classification model for Task A (Gender Classification).
- [`face_recognition_model.pth`](https://drive.google.com/file/d/1Xkwl3xrfl2MUC5zJWC9XZRfP3pOuwvBR/view?usp=sharing) – Embedding-based model backbone for Task B (Face Recognition).


## 📂 Folder Structure

```
facecom-hackathon/
│
├── facecom_dataset/
│   ├── Comys_Hackathon5/         # Raw dataset
│   │   └── Task_A/, Task_B/
│   ├── processed_gender/         # Cropped faces for gender classification
│   ├── processed_faces/          # Cropped faces for face recognition
│   ├── Task_A/
│   │   └── facecom_task_a.py     # Gender classification script
│   └── Task_B/
│       └── facecom_task_b.py     # Face recognition script
│
├── gender_model.pth              # Saved model (Task A)
├── face_recognition_model.pth    # Saved model (Task B)
├── task_b_results.csv            # Inference results (Task B)
└── README.md
```

---

## 🙋‍♀️ Author

**Tanisha Debnath** **Ishika Dutta**
B.Tech CSE (AI) | Institute of Engineering and Management, Kolkata
B.Tech CSE | Institute of Engineering and Management, Kolkata
🔗 [Portfolio](https://tanisha-debnath-portfolio.web.app) • [GitHub](https://github.com/TanishaDebnath)

---

## 🏁 Acknowledgements

* **Comsys Hackathon Team**
* **Jadavpur University**
* All open-source contributors to PyTorch, OpenCV, and scikit-learn

```

