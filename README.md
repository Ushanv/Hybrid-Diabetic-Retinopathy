# Hybrid Diabetic Retinopathy Detection

This repository contains a hybrid deep learning approach for the detection of Diabetic Retinopathy (DR) from retinal fundus images. The project combines Convolutional Neural Networks (CNNs) and traditional machine learning models to enhance classification accuracy and improve early detection of DR, which is crucial in preventing vision loss.

# Project Objective

The main objective of this project is to develop a robust and accurate hybrid model that can classify the severity of Diabetic Retinopathy in retinal images into multiple stages, leveraging both deep learning and classical machine learning techniques.

# 📁 Dataset

The model was trained and evaluated using publicly available datasets, such as:

- **APTOS 2019 Blindness Detection** – [Kaggle Link](https://www.kaggle.com/c/aptos2019-blindness-detection)
- **EyePACS Dataset** – A large dataset of high-resolution retina images, labeled by severity.

Each image is categorized into one of five classes:

1. No DR
2. Mild
3. Moderate
4. Severe
5. Proliferative DR

## 🧠 Model Architecture

The hybrid model integrates:

- **CNN-based feature extractor**: Used to learn spatial patterns and deep features from the fundus images.
- **Machine Learning Classifier**: Extracted features are then fed into classifiers like Random Forest, XGBoost, or SVM for final classification.

### Key Components:

- Image pre-processing (resizing, normalization, contrast enhancement)
- Feature extraction via pre-trained CNN models (e.g., VGG16, ResNet50)
- Feature reduction techniques (PCA, t-SNE)
- Final classification using traditional ML algorithms

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- OpenCV
- Matplotlib / Seaborn

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Ushanv/Hybrid-Diabetic-Retinopathy.git
   cd Hybrid-Diabetic-Retinopathy
2. pip install -r requirements.txt
3. python main.py

# Results
Model	              Accuracy	Precision	Recall	F1 Score
CNN Only	          82.4%	    80.2%	    81.1%	  80.6%
Hybrid (CNN + ML)	  88.7%	    87.1%	    88.0%	  87.5%
