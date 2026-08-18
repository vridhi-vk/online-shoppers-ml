# Online Shoppers Purchase Intention Prediction

## a. Problem Statement

The objective of this project is to predict whether an online shopping session will result in a purchase using machine learning classification algorithms.

Five classification models were implemented and compared:

- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Random Forest Classifier

The models were evaluated using Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

## b. Dataset Description

Dataset: Online Shoppers Purchasing Intention Dataset

Source: UCI Machine Learning Repository

The dataset contains 12,330 online shopping sessions. It contains 17 predictor features and one target variable, Revenue.

The features contain information about user browsing behavior, including:
- Number and duration of administrative pages visited
- Number and duration of informational pages visited
- Number and duration of product-related pages visited
- Bounce rate
- Exit rate
- Page value
- Special day information
- Month
- Operating system
- Browser
- Region
- Traffic type
- Visitor type
- Weekend indicator

Target variable:

Revenue = True indicates that the session resulted in a purchase.

Revenue = False indicates that the session did not result in a purchase.

The dataset is imbalanced, with substantially more non-purchasing sessions than purchasing sessions.

## c. GitHub Repository Link

[https://github.com/vridhi-vk/online-shoppers-ml]

## d. Models Used and Evaluation Results

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8808 | 0.8828 | 0.7366 | 0.3586 | 0.4824 | 0.4592 |
| Decision Tree | 0.8585 | 0.7356 | 0.5420 | 0.5576 | 0.5497 | 0.4658 |
| KNN | 0.8767 | 0.7633 | 0.6822 | 0.3822 | 0.4899 | 0.4493 |
| Naive Bayes | 0.2729 | 0.7334 | 0.1726 | 0.9738 | 0.2933 | 0.1289 |
| Random Forest | 0.8970 | 0.9131 | 0.7712 | 0.4764 | 0.5890 | 0.5541 |

## Model Performance Observations

### Logistic Regression
Logistic Regression achieved good Accuracy and AUC and relatively high Precision. However, its Recall was low, indicating that it failed to identify many purchasing sessions.

### Decision Tree
The Decision Tree achieved lower Accuracy and AUC than Logistic Regression, but it obtained better Recall. Its F1 Score and MCC indicate reasonably balanced classification performance.

### K-Nearest Neighbors
KNN achieved good Accuracy and Precision but relatively low Recall. Its overall performance was lower than Logistic Regression and Random Forest.

### Naive Bayes
Naive Bayes achieved very high Recall, detecting almost all purchasing sessions. However, its very low Precision and Accuracy indicate a large number of false positive predictions.

### Random Forest
Random Forest achieved the highest Accuracy, AUC, Precision, F1 Score, and MCC among the evaluated models. It provided the best overall balance between identifying purchasing sessions and limiting incorrect positive predictions.

### Overall Winner
Random Forest was selected as the best-performing model because it achieved the strongest overall performance across the evaluation metrics.

## Streamlit Application

The Streamlit application allows users to:

- Upload test data in CSV format
- Select one of the five classification models
- View Accuracy, AUC, Precision, Recall, F1, and MCC
- View the confusion matrix
- View the classification report

## Live Streamlit App

[https://online-shoppers-ml-dvbls8spw8pvfs4jbhxfjb.streamlit.app/]