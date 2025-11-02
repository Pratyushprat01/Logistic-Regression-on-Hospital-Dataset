❤ Heart Disease Prediction using Machine Learning
📘 Overview

This project aims to predict the 10-year risk of heart disease based on various medical and lifestyle factors.
Using machine learning and exploratory data analysis (EDA), the goal was to uncover health patterns and build a model that can help in early risk identification.

🎯 Problem Statement

Heart disease remains a leading cause of mortality worldwide.
The objective of this project is to use clinical and behavioral attributes such as age, BMI, cholesterol, blood pressure, and smoking habits to predict whether a person is at risk of developing heart disease within the next 10 years.

🧠 Key Challenges

Imbalanced Dataset — The number of people with heart disease was much lower than those without.

Feature Scaling — Variables like glucose, cholesterol, and blood pressure had different scales, which affected model performance.

Data Quality — Missing values and outliers needed careful treatment to ensure data reliability.

⚙ Tech Stack

Language: Python

Libraries:

pandas, numpy → Data cleaning & manipulation

matplotlib, seaborn → Data visualization & outlier detection

scikit-learn → Model building & evaluation

Algorithm: Logistic Regression (class_weight='balanced')

🔍 Approach
1. Data Preprocessing

Handled missing values and converted categorical data where necessary.

Detected and visualized outliers using boxplots.

Converted incorrect data types (BMI, glucose, heartRate) to numeric.

2. Exploratory Data Analysis (EDA)

Analyzed relationships between risk factors and heart disease using pairplots and correlation heatmaps.

Visualized the distribution of features like BMI, glucose, smoking habits, etc.

3. Feature Scaling

Used StandardScaler to normalize features for better model convergence.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

4. Model Building

Implemented Logistic Regression with class_weight='balanced' to handle the class imbalance effectively.

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

5. Evaluation

Evaluated the model using accuracy, precision, recall, and F1-score to ensure fair assessment, especially for the minority (heart disease) class.

📊 Results

The model achieved a balanced performance between accuracy and recall.

It showed improved ability to identify positive (heart disease) cases after balancing the dataset.

While overall accuracy was moderate, the recall for high-risk cases improved — which is more important in healthcare applications.

💡 Key Learnings

Accuracy alone is not sufficient for imbalanced datasets — recall and precision are equally crucial.

Scaling and balancing techniques can drastically improve model fairness.

Logistic Regression, though simple, is highly interpretable and effective for healthcare risk prediction.
