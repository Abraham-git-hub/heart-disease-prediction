# Heart Disease Prediction Using Machine Learning

A supervised learning project that predicts the presence of heart disease based on common medical test results, comparing multiple machine learning algorithms to identify the optimal approach for early detection.

## 📋 Project Overview

Heart disease remains the leading cause of death globally, claiming approximately 17.9 million lives annually. This project develops machine learning models to predict heart disease presence based on patient medical data, aiming to create an accessible screening tool that could assist healthcare providers in early detection and potentially save lives through timely intervention.

### Goals

1. Develop accurate classification models to predict heart disease presence
2. Identify which medical factors are most predictive of heart disease  
3. Compare multiple machine learning algorithms to find the optimal approach
4. Create interpretable results that could inform clinical decision-making

## 🎓 Course Information

- **Course**: Supervised Machine Learning
- **Semester**: Fall 2025
- **Author**: Abraham Asseffa

## 📊 Dataset

**Source**: [Heart Disease Cleveland UCI Dataset](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)

**Citation**:
```
Cherngs. (2020). Heart Disease Cleveland UCI [Data set]. Kaggle. 
https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci

Original Source: UCI Machine Learning Repository
Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). 
Heart Disease Data Set. UCI Machine Learning Repository.
```

**Dataset Details**:
- **Size**: 297 patients × 14 features
- **Type**: Supervised binary classification
- **Target**: Presence (1) or absence (0) of heart disease
- **Class Balance**: 54% no disease, 46% disease

### Features

| Feature | Description | Type |
|---------|-------------|------|
| age | Age in years | Continuous |
| sex | Gender (1=male, 0=female) | Binary |
| cp | Chest pain type (0-3) | Categorical |
| trestbps | Resting blood pressure (mm Hg) | Continuous |
| chol | Serum cholesterol (mg/dl) | Continuous |
| fbs | Fasting blood sugar > 120 mg/dl | Binary |
| restecg | Resting ECG results (0-2) | Categorical |
| thalach | Maximum heart rate achieved | Continuous |
| exang | Exercise induced angina | Binary |
| oldpeak | ST depression induced by exercise | Continuous |
| slope | Slope of peak exercise ST segment | Categorical |
| ca | Number of major vessels (0-3) | Categorical |
| thal | Thalassemia (0-2) | Categorical |
| condition | **Target**: Heart disease diagnosis | Binary |

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Data quality assessment (missing values, duplicates, outliers)
- Feature distribution analysis
- Correlation analysis between features
- Target variable balance check
- Comprehensive visualizations

### 2. Data Preparation
- Train-test split (80-20) with stratification
- Feature scaling using StandardScaler
- No missing values or significant data quality issues

### 3. Models Implemented

| Model | Hyperparameter Tuning | Key Parameters |
|-------|----------------------|----------------|
| **Logistic Regression** | No | max_iter=1000 |
| **Random Forest** | ✅ GridSearchCV | n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features |
| **Support Vector Machine (SVM)** | ✅ GridSearchCV | C, kernel, gamma |
| **K-Nearest Neighbors (KNN)** | No | n_neighbors=5 |

### 4. Evaluation Metrics
- Accuracy
- Precision  
- Recall (critical for medical diagnosis)
- F1-Score
- ROC-AUC curves
- Confusion matrices
- Feature importance analysis

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~0.87 | ~0.86 | ~0.89 | ~0.87 |
| Random Forest (tuned) | **~0.90** | **~0.89** | **~0.91** | **~0.90** |
| SVM (tuned) | ~0.88 | ~0.87 | ~0.90 | ~0.88 |
| KNN | ~0.85 | ~0.84 | ~0.87 | ~0.85 |

### Key Findings

1. **Random Forest with hyperparameter tuning achieved the best overall performance**
2. **Most Important Predictive Features**:
   - Chest pain type (cp)
   - Maximum heart rate achieved (thalach)  
   - ST depression (oldpeak)
   - Exercise induced angina (exang)
   - Number of major vessels (ca)

3. **Hyperparameter tuning significantly improved model performance** (3-5% improvement)
4. All models achieved >85% accuracy, demonstrating strong predictive capability

## 🗂️ Repository Structure

```
heart-disease-prediction/
│
├── Heart_Disease_Prediction.ipynb    # Main Jupyter notebook with full analysis
├── heart_cleveland_upload.csv        # Dataset (place here to run notebook)
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
└── .gitignore                        # Git ignore file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

1. Clone this repository:
```bash
git clone https://github.com/Abraham-git-hub/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci) and place `heart_cleveland_upload.csv` in the project root directory.

4. Launch Jupyter Notebook:
```bash
jupyter notebook
```

5. Open `Heart_Disease_Prediction.ipynb` and run all cells (Kernel → Restart & Run All)

### Required Python Packages

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

## 📊 Visualizations

The notebook includes comprehensive visualizations:

- **Data Quality**: Missing values, outliers, distributions
- **EDA**: Feature distributions, correlations, pairplots
- **Model Performance**: Confusion matrices, ROC curves, feature importance
- **Comparisons**: Side-by-side model performance metrics

## 💡 Key Insights

### What Worked Well

✅ Hyperparameter tuning via GridSearchCV significantly improved performance  
✅ Feature scaling was crucial for SVM and KNN performance  
✅ Cross-validation ensured robust model evaluation  
✅ Ensemble methods (Random Forest) outperformed simpler models  

### Limitations

⚠️ Small dataset (297 patients) - larger dataset would improve generalization  
⚠️ Limited feature engineering performed  
⚠️ Model not validated on external dataset from different hospital  
⚠️ Computational cost of grid search limited parameter exploration  

## 🔮 Future Work

1. **Data Collection**: Gather more data from diverse populations and hospitals
2. **Feature Engineering**: Create interaction terms, polynomial features
3. **Advanced Models**: Implement XGBoost, LightGBM, Neural Networks
4. **Interpretability**: Add SHAP values, LIME explanations
5. **Deployment**: Create web application for healthcare providers
6. **External Validation**: Test on independent dataset before clinical use
7. **Additional Features**: Include family history, lifestyle factors

## 🏥 Clinical Implications

**Important**: These models are intended as screening tools to flag high-risk patients, **not to replace expert medical diagnosis**. 

- High recall is critical in medical contexts to minimize false negatives
- Feature importance aligns with established medical knowledge
- Could assist healthcare providers in early detection strategies
- Requires validation and regulatory approval before clinical deployment

## 📚 References

1. Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). Heart Disease Data Set. UCI Machine Learning Repository.

2. Cherngs. (2020). Heart Disease Cleveland UCI [Data set]. Kaggle. https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci

3. World Health Organization. (2021). Cardiovascular diseases (CVDs). https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)

## 📹 Video Presentation

[Link to video presentation will be added here]

## 📄 License

This project is for educational purposes as part of a university course. The dataset is publicly available from Kaggle and the UCI Machine Learning Repository.

## 👤 Author

**Abraham Asseffa**

- GitHub: [@Abraham-git-hub](https://github.com/Abraham-git-hub)
- Project Link: [https://github.com/Abraham-git-hub/heart-disease-prediction](https://github.com/Abraham-git-hub/heart-disease-prediction)

## 🙏 Acknowledgments

- Cleveland Clinic Foundation for collecting and sharing this valuable dataset
- UCI Machine Learning Repository for maintaining the dataset
- Course instructors and peers for feedback and support

---

**⭐ If you found this project helpful, please consider giving it a star!**
