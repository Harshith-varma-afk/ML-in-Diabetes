# Predicting Diabetes Progression via Machine Learning

This project investigates the application of various machine learning models to predict diabetes status based on personal health indicators. Utilizing a large-scale public health dataset, we address severe class imbalance and evaluate the trade-offs between linear and non-linear classification methods for early disease detection.

## 📌 Project Overview

* **Domain:** Public Health / Predictive Healthcare
* **Dataset:** CDC Diabetes Health Indicators (extracted from the Behavioral Risk Factor Surveillance System - BRFSS)
* **Data Size:** 253,680 instances with 21 attributes (e.g., BMI, age, smoking status, physical activity)
* **Target Classes:** Non-diabetic (0), Pre-diabetic (1), and Diabetic (2)

---

## ⚙️ Methodology & Pipeline

The pipeline was designed to handle high dimensionality, missing data, and severe class imbalance before training models across multiple random train-test splits.

### 1. Data Preprocessing

* **Imputation:** Missing values were handled using **mean substitution**.
* **Encoding:** Categorical features were converted using **one-hot encoding**.
* **Scaling:** Numerical features were normalized via **Z-score scaling** to ensure feature comparability.

### 2. Class Imbalance Mediation

Because the dataset is heavily skewed toward healthy individuals, **SMOTENC** (Synthetic Minority Over-sampling Technique for Nominal and Continuous) was applied within cross-validation loops to synthetically balance the training sets.

### 3. Dimensionality Reduction

**Principal Component Analysis (PCA)** was performed post-scaling, preserving **95% of the variance** to reduce noise and enhance computational efficiency.

### 4. Classification Strategy

The multi-class problem was broken down into two distinct binary classification tasks:

* **Task 1:** Non-diabetic vs. At-Risk (Pre-diabetic or Diabetic lumped together).
* **Task 2:** Pre-diabetic vs. Diabetic (with non-diabetic records completely excluded).

---

## 🤖 Models & Hyperparameter Tuning

Four distinct algorithms were trained and optimized using **GridSearchCV** or **RandomizedSearchCV** with stratified k-fold cross-validation:

* **Logistic Regression:** Evaluated across different sample sizes (100k and 150k); tuned for regularization strength ($C$), penalty type ($L_1$/$L_2$), and the `liblinear` solver.
* **Multilayer Perceptron (MLP):** Evaluated on a 100k split due to computational intensity. Handled 2 hidden layers, optimized for hidden layer neurons ($N \in \{10, 15, 25\}$) and $L_2$ regularization ($\alpha$).
* **Kernel SVM (RBF Kernel):** Evaluated using default hyperparameters on stratified, down-sampled subsets (max 10,000 points) for computational viability.
* **Random Forest:** Tuned for the number of estimators, tree depth, and minimum node split/leaf sizes while leveraging balanced class weights.

---

## 📊 Key Results Summary

| Model | Task 1: Healthy vs. At-Risk | Task 2: Pre-Diabetic vs. Diabetic | Key Takeaways & Performance Nuances |
| --- | --- | --- | --- |
| **Logistic Regression** | **81–82%** Accuracy | **78%** Accuracy | Consistently lowest performance; struggled with non-linearly separable decision boundaries but was the fastest to train. |
| **Multilayer Perceptron** | **84–85%** Accuracy | **77–78%** Accuracy | High success rate identifying non-healthy individuals (**90–91%**); strong, balanced sensitivity/specificity (~81% / ~75%) on Task 2. |
| **Random Forest** | **~92%** Accuracy | **~87–88%** Accuracy | High accuracy and exceptional specificity (**~97–98%**), but suffered from critically low sensitivity (**5–8%** for diabetics, **28–29%** for pre-diabetics). |
| **Kernel SVM (RBF)** | **~78%** Accuracy | **68%** Accuracy | Moderate performance; achieved 81% sensitivity and 74% specificity on Healthy vs. Diabetic subsets. |

---

## 💡 Key Discussion Points

* **Linear vs. Non-Linear:** Logistic Regression performed poorly because the target boundaries are highly non-linear. Non-linear models (MLP, RF, SVM) captured complex trends much better, though they required significantly higher computational costs.
* **The Sensitivity Challenge:** While models achieved great overall accuracy and specificity, identifying true positive cases (especially capturing the minority pre-diabetic class) remains difficult.
* **SMOTE Limitations:** Synthetic data quality inherently drops when the base minority class (pre-diabetics) is extremely small to begin with, making it tough for SMOTE to generate high-quality boundaries.

## 🚀 Future Work

* Test alternative resampling strategies (e.g., balance via advanced ensemble methods like EasyEnsemble or BalancedRandomForest).
* Perform deeper hyperparameter optimization on the Kernel SVM model.
* Incorporate more diverse or complete real-world healthcare datasets to enhance model generalizability.
