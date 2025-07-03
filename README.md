
````markdown
# 🧠 Facecom Hackathon 2025 – Robust Face & Gender Recognition

This repository contains my solution for the **Facecom Hackathon 2025** conducted by **Jadavpur University**, focused on robust face recognition and gender classification under adverse visual conditions.

---

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

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![TQDM](https://img.shields.io/badge/TQDM-FF6F00?style=for-the-badge)](https://tqdm.github.io/)
[![Pillow](https://img.shields.io/badge/Pillow-3693F3?style=for-the-badge&logo=python&logoColor=white)](https://python-pillow.org/)

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

---

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

**Tanisha Debnath**
B.Tech CSE (AI) | Institute of Engineering and Management, Kolkata
🔗 [Portfolio](https://tanisha-debnath-portfolio.web.app) • [GitHub](https://github.com/TanishaDebnath)

---

## 🏁 Acknowledgements

* **FACECOM Hackathon Team**
* **Jadavpur University**
* All open-source contributors to PyTorch, OpenCV, and scikit-learn

```

---


```
