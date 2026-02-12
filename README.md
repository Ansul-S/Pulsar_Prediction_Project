# 🌟 Pulsar Star Classification using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An end-to-end machine learning pipeline to identify pulsar stars from radio telescope data, achieving **92% recall** and minimizing false negatives to maximize astronomical discoveries.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Key Learnings](#key-learnings)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Future Improvements](#future-improvements)
- [Contact](#contact)

---

## 🔭 Overview

Pulsars are highly magnetized rotating neutron stars that emit beams of electromagnetic radiation. With only ~3,000 pulsars discovered out of an estimated 100,000 in our galaxy, automated detection is crucial for accelerating astronomical research.

This project builds a machine learning classifier to identify pulsar candidates from radio telescope data, **reducing manual review workload by 98%** while maintaining **92% detection accuracy**.

### 🎯 Key Achievements

- ✅ **92% Recall** - Successfully detects 284 out of 308 pulsars
- ✅ **24 False Negatives** - Lowest miss rate across all tested models
- ✅ **96.68% Accuracy** - Strong overall performance
- ✅ **Handles Severe Class Imbalance** - 91:9 negative-to-positive ratio

---

## 🎯 Problem Statement

### Why This Matters

Pulsars provide insights into:
- Extreme physics (neutron stars, gravitational waves)
- Tests of general relativity
- Deep space navigation systems
- Understanding stellar evolution

### The Challenge

1. **Severe Class Imbalance**: Only 9.16% of candidates are actual pulsars
2. **High-Stakes Decisions**: Missing a pulsar = lost scientific discovery
3. **Asymmetric Cost**: False negatives (missed discoveries) are worse than false positives (false alarms)

### Solution Approach

Build a machine learning classifier that **prioritizes recall** (maximizing pulsar detection) while maintaining acceptable precision (minimizing wasted telescope follow-up time).

---

## 📊 Dataset

**Source**: HTRU2 Dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/HTRU2)

### Specifications

- **Size**: 17,898 observations
- **Features**: 8 continuous variables
- **Target**: Binary (0 = Not Pulsar, 1 = Pulsar)
- **Class Distribution**: 
  - Not Pulsar: 16,259 (90.84%)
  - Pulsar: 1,639 (9.16%)
  - Imbalance Ratio: ~10:1

### Features

**Integrated Profile Statistics:**
- Mean, Standard Deviation, Kurtosis, Skewness of integrated pulse profile

**DM-SNR Curve Statistics:**
- Mean, Standard Deviation, Kurtosis, Skewness of DM-SNR curve

---

## 🔬 Methodology

### Phase 1: Exploratory Data Analysis

- Statistical analysis and distribution checks
- Correlation analysis and feature relationships
- Outlier detection and class imbalance assessment
- Visualization of feature separability

**Key Finding**: Kurtosis features capture pulse shape characteristics critical for classification.

### Phase 2: Data Preprocessing

#### Outlier Handling
**Decision**: Retained outliers  
**Rationale**: Pulsars are extreme objects; their "outlier" values represent real astronomical signals, not measurement errors.

#### Feature Scaling
**Method**: RobustScaler  
**Rationale**: Uses median and IQR, unaffected by outliers (unlike StandardScaler which uses mean/std).
````python
# Proper train-test split to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit scaler on training data only
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use training parameters
````

#### Class Imbalance Handling

**Approaches Tested**:

1. **SMOTE (Synthetic Minority Over-sampling)**
   - Created synthetic minority samples
   - Balanced training set to 50:50 ratio
   - Result: Good baseline performance

2. **Class Weights** (Final Choice) ✅
   - Used XGBoost's `scale_pos_weight` parameter
   - Trained on original imbalanced data with weighted loss
   - Result: **Best performance** (24 false negatives)
````python
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
# ≈ 9.76 - tells XGBoost to treat minority class as 9.76x more important
````

### Phase 3: Model Development

Systematically evaluated multiple algorithms:

#### 1. Logistic Regression (Baseline)
- **Purpose**: Establish minimum performance benchmark
- **Result**: 96.76% accuracy, 91% recall, 29 FN
- **Conclusion**: Strong baseline, room for improvement

#### 2. Random Forest
- **Purpose**: Capture non-linear relationships and feature interactions
- **Result**: 97.79% accuracy (highest), 90% recall, 30 FN
- **Notable**: Revealed Kurtosis_Integrated as most important feature (33%)

#### 3. XGBoost (Final Model) ⭐
- **Purpose**: State-of-the-art gradient boosting with class imbalance handling
- **Result**: 96.70% accuracy, **92% recall (highest)**, **24 FN (lowest)**
- **Decision**: Selected for minimizing false negatives

#### Hyperparameter Tuning Experiment

Used RandomizedSearchCV to optimize XGBoost:
- **Unexpected Result**: Tuning increased false negatives (27→31)
- **Decision**: Retained simpler model with better FN performance
- **Learning**: Default parameters often well-designed; blind tuning can hurt target metrics

---

## 📈 Results

### Final Model Performance
````
Model: XGBoost with Class Weights
════════════════════════════════════════

Accuracy:  96.70%
Recall:    92.21% ⭐ (Highest)
Precision: 74.93%
F1-Score:  82.75%

Confusion Matrix:
                 Predicted
              Not Pulsar  |  Pulsar
Actual ──────────────────┼──────────
Not Pulsar    3178  ✅   |   94  ❌
Pulsar          24  ❌   |  284  ✅

False Negatives: 24 (7.79% of pulsars missed)
False Positives: 95 (acceptable for discovery science)
````

### Model Comparison

| Model | Accuracy | Recall | Precision | False Negatives |
|-------|----------|--------|-----------|-----------------|
| Logistic Regression | 96.76% | 91% | 76% | 29 |
| Random Forest | **97.79%** | 90% | **85%** | 30 |
| **XGBoost (Final)** | 96.70% | **92%** | 75% | **24** ⭐ |

### Business Impact

**Without ML**:
- Manual review of 17,898 candidates
- Time-consuming and error-prone
- Potential missed pulsars due to fatigue

**With ML System**:
- Flags 379 high-probability candidates (98% reduction)
- 284 true pulsars + 94 false alarms
- Only 24 pulsars missed (7.79%)
- **Result**: 98% workload reduction, 92% detection rate

---

## 💡 Key Learnings

### Technical Insights

1. **Class Imbalance**: `scale_pos_weight` outperformed SMOTE for XGBoost
2. **Feature Engineering**: Model-based importance differed from visual EDA
3. **Scaling**: RobustScaler essential for astronomical data with outliers
4. **Data Leakage**: Always split before preprocessing
5. **Evaluation**: Accuracy misleading for imbalanced data; recall critical

### Methodological Insights

1. **Hyperparameter Tuning**: Doesn't always improve target metrics
2. **Multiple Approaches**: SMOTE vs class weights - context determines best choice
3. **Domain Knowledge**: False negatives worse than false positives in discovery science
4. **Model Selection**: Best model ≠ highest accuracy; depends on business objective

---

## 🚀 Installation

### Prerequisites
````bash
Python 3.8+
````

### Setup

1. **Clone the repository**
````bash
git clone https://github.com/[YOUR_USERNAME]/pulsar-star-classification.git
cd pulsar-star-classification
````

2. **Install dependencies**
````bash
pip install -r requirements.txt
````

Or install manually:
````bash
pip install numpy pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn
````

3. **Download dataset**
````python
# Automatically downloads HTRU2 dataset
python download_data.py
````

---

## 💻 Usage

### Training the Model
````python
# Run complete pipeline
python train_model.py

# Or step-by-step in Jupyter
jupyter notebook notebooks/pulsar_classification.ipynb
````

### Making Predictions
````python
import pickle
import numpy as np

# Load model and scaler
with open('models/xgboost_final.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare new data
new_data = np.array([[140.5, 55.7, -0.23, -0.70, 3.2, 19.1, 8.0, 74.2]])
scaled_data = scaler.transform(new_data)

# Predict
prediction = model.predict(scaled_data)
probability = model.predict_proba(scaled_data)

print(f"Prediction: {'Pulsar' if prediction[0] == 1 else 'Not Pulsar'}")
print(f"Confidence: {probability[0][prediction[0]]:.2%}")
````

---

## 📁 Project Structure
````
pulsar-star-classification/
│
├── data/
│   ├── raw/                    # Original HTRU2 dataset
│   └── processed/              # Preprocessed data
│
├── notebooks/
│   ├── 01_EDA.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb # Data preprocessing
│   ├── 03_modeling.ipynb      # Model training & comparison
│   └── 04_evaluation.ipynb    # Final evaluation & analysis
│
├── src/
│   ├── data_preprocessing.py  # Preprocessing functions
│   ├── model_training.py      # Model training utilities
│   ├── evaluation.py          # Evaluation metrics
│   └── utils.py               # Helper functions
│
├── models/
│   ├── xgboost_final.pkl      # Final trained model
│   └── scaler.pkl             # Fitted RobustScaler
│
├── visualizations/
│   ├── eda_plots/             # EDA visualizations
│   ├── model_comparison/      # Model performance plots
│   └── confusion_matrices/    # Confusion matrices
│
├── requirements.txt           # Python dependencies
├── train_model.py            # Training script
├── download_data.py          # Dataset download script
├── README.md                 # This file
└── LICENSE                   # MIT License
````

---

## 🛠️ Technologies

**Core Libraries**:
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Scikit-learn** - ML algorithms, preprocessing, evaluation
- **XGBoost** - Gradient boosting implementation
- **imbalanced-learn** - SMOTE and imbalance handling

**Visualization**:
- **Matplotlib** - Basic plotting
- **Seaborn** - Statistical visualizations

**Development**:
- **Jupyter** - Interactive development
- **Pycharm** - Local IDE
- **Git/GitHub** - Version control

---

## 🔮 Future Improvements

### Short-term
- [ ] Ensemble voting (combine Logistic Regression + RF + XGBoost)
- [ ] SHAP values for model interpretability
- [ ] Streamlit web app for easy predictions

### Long-term
- [ ] Deep learning approach (1D CNN on raw signal data)
- [ ] Anomaly detection framework (Isolation Forest)
- [ ] Real-time prediction API
- [ ] Integration with telescope data pipelines

---

## 📚 References

- R. J. Lyon et al. (2016). "Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach." Monthly Notices of the Royal Astronomical Society.
- Dataset: [UCI Machine Learning Repository - HTRU2](https://archive.ics.uci.edu/ml/datasets/HTRU2)
- XGBoost Documentation: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)

---

## 📧 Contact

**[Ansul Suryawanshi]**  
📧 Email: [ansul2612@gmail.com]  
💼 LinkedIn: [Ansul Suryawanshi](https://www.linkedin.com/in/ansul-suryawanshi-a365ab15a/)
🐙 GitHub: [Ansul-S](https://github.com/Ansul-S)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Acknowledgments

- **Dataset**: HTRU2 from UCI Machine Learning Repository
- **Inspiration**: Real-world astronomical discovery challenges
- **Learning**: Hands-on project-based learning approach

---

⭐ **If you found this project helpful, please consider giving it a star!**

---

**Built with ❤️ and lots of efforts over 4 days**
````
