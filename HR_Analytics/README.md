# 💼 HR Employee Attrition Prediction

This project uses **Logistic Regression** and **Random Forest Classifier** to predict whether an employee is likely to leave the company based on HR data.

---

## 📂 Dataset

- **Source**: IBM HR Analytics Employee Attrition & Performance
- **Rows**: 1470  
- **Target Column**: `Attrition` (Yes/No)
- **Features**: Demographic, job-related, and performance features such as `Age`, `JobRole`, `MonthlyIncome`, `JobSatisfaction`, etc.

---

## 🔍 Problem Statement

Employee attrition (resignation/turnover) can be costly and disruptive for organizations. By analyzing historical HR data, we aim to build a predictive model that can identify employees who are at risk of leaving.

---

## 🧪 Libraries Used

- `pandas` for data loading & manipulation  
- `sklearn` for preprocessing, modeling, and evaluation  
- `LabelEncoder` and `OneHotEncoder` for categorical encoding  
- `StandardScaler` for feature scaling  
- `train_test_split` to divide the dataset  
- `classification_report`, `confusion_matrix`, and `accuracy_score` for model evaluation  

---

## ⚙️ Preprocessing Steps

1. **Dropped irrelevant columns**:  
   - `EmployeeNumber`, `EmployeeCount`, `Over18`, `StandardHours`

2. **Categorical Encoding**:
   - **Binary features** (e.g., `Gender`, `OverTime`) → Label Encoding
   - **Multi-class features** (e.g., `Department`, `JobRole`) → One-Hot Encoding

3. **Feature Scaling**:
   - Applied `StandardScaler` on all numerical features for normalization

4. **Train-Test Split**:
   - 80% Training, 20% Testing  
   - Stratified by target to maintain class balance

---

## 🧠 Models Used

### 1. **Logistic Regression**
- Simple baseline linear model
- Accuracy: **86.05%**
- Better recall for identifying employees who actually left

### 2. **Random Forest Classifier**
- Ensemble model with decision trees
- Accuracy: **82.99%**
- Performed worse in recall for leavers (due to class imbalance)

---

## 📊 Model Evaluation

| Metric             | Logistic Regression | Random Forest |
|-------------------|---------------------|---------------|
| **Accuracy**       | 86.05%              | 82.99%        |
| **Recall (Leavers)** | 34%                | 9%            |
| **Precision (Leavers)** | 62%            | 36%           |
| **F1-score (Leavers)** | 44%             | 14%           |

> ⚠️ Note: The dataset is **imbalanced**, with far fewer `Attrition = Yes` records, causing lower recall in Random Forest.

---
## 🧾 Output Snippet


Logistic Regression Accuracy: 0.8605442176870748
Random Forest Accuracy: 0.8299319727891157

--- Logistic Regression Report ---
              precision    recall  f1-score   support

           0       0.88      0.96      0.92       247
           1       0.62      0.34      0.44        47

    accuracy                           0.86       294
   macro avg       0.75      0.65      0.68       294
weighted avg       0.84      0.86      0.84       294


--- Random Forest Report ---
              precision    recall  f1-score   support

           0       0.85      0.97      0.91       247
           1       0.36      0.09      0.14        47

    accuracy                           0.83       294
   macro avg       0.61      0.53      0.52       294
weighted avg       0.77      0.83      0.78       294

Confusion Matrix (LogReg):
 [[237  10]
 [ 31  16]]
Confusion Matrix (RF):
 [[240   7]
 [ 43   4]]
---
## 🙋‍♀️ Author

**Keerthana R**  
Python | Machine Learning Enthusiast   
🔗 [GitHub Profile](https://github.com/keerthanas0714/AI-Projects)