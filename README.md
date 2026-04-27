# 🌿 AI Based Leafy Vegetable Disease Detection

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56-red)
![Accuracy](https://img.shields.io/badge/Accuracy-87%25-green)

## 📌 Overview
An AI powered web application that automatically detects diseases 
in leafy vegetables using EfficientNet-B3 deep learning model with 
Transfer Learning. The system provides disease name, confidence score, 
cause, remedy and crop quality prediction.

---

## 🥬 Vegetables Covered
| Vegetable | Disease Classes |
|---|---|
| Cabbage | Alternia Leaf Spot, Aphid Colony, Ring Spot, Healthy |
| Malabar Spinach | Anthracnose, Bacterial Spot, Pest Damage, Healthy |
| Potato | Early Blight, Later Blight, Healthy |
| Radish | Black Leaf Spot, Flea Beetle, Mosaic Virus, Healthy |

---

## 🧠 Model Details
| Parameter | Value |
|---|---|
| Model | EfficientNet-B3 |
| Framework | PyTorch |
| Training | Transfer Learning + Two Phase Training |
| Phase 1 | 10 Epochs, LR: 0.001 |
| Phase 2 | 20 Epochs, LR: 0.0001 (Fine Tuning) |
| Accuracy | 87%+ |
| Test Confidence | 96% - 100% |
| Total Classes | 15 |

---

## 🚀 How to Run

### Step 1 — Clone Repository
```bash
git clone https://github.com/abhishekji549/Plant-Disease-Detection.git
cd Plant-Disease-Detection
```

### Step 2 — Install Libraries
```bash
pip install streamlit torch torchvision pillow
```

### Step 3 — Download Trained Model
👉 [Download plant_disease_model.pth](https://drive.google.com/file/d/1MzsXkr9Gq6Vc9kFUTCZGBvPzj-Pl1LgF/view?usp=drive_link)

Place the downloaded file in the project root folder.

### Step 4 — Run App
```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack
- Python 3.11
- PyTorch
- EfficientNet-B3 (ImageNet Pretrained)
- Streamlit
- Google Colab T4 GPU
- VS Code

---

## 📊 Results
- ✅ 87%+ Validation Accuracy
- ✅ 96-100% Test Confidence
- ✅ All 15 classes correctly classified
- ✅ No overfitting detected
- ✅ Real-time disease diagnosis

---

## 👨‍💻 Team
| Name | Role |
|---|---|
| Abhishek Yadav | Model Training & Streamlit App  |
| Anurag Kori | Report Making  |
| Aditya Ben | Research  |
| Himanshu Meshram | Support and Ppt |
| Kartik Malik | Research & Support |

---

## 🏫 College
**Jabalpur Engineering College**  
Department: Artificial Intelligence & Data Science  
Academic Year: 2025-26

---

## 📄 References
- Mohanty et al. (2016) — Plant Disease Detection using CNN
- Atila et al. (2021) — EfficientNet for Plant Disease Classification
- PlantVillage Dataset — Kaggle
