# Subject Classification Project

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Data Collection and Preprocessing](#data-collection-and-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Machine Learning Models](#machine-learning-models)
- [Results and Evaluation](#results-and-evaluation)
- [Key Findings](#key-findings)
- [Future Work](#future-work)
- [Technologies Used](#technologies-used)

## Introduction

This project focuses on developing a machine learning model to classify text into different academic subjects. The goal is to accurately categorize text samples into subjects such as Biology, Mathematics, Physics, Chemistry, History, Geography, and Political Science.

## Project Overview

The project consists of several key components:
1. Web scraping to collect subject-specific text data
2. Data preprocessing and cleaning
3. Exploratory data analysis and visualization
4. Implementation of various machine learning models for text classification
5. Model evaluation and comparison

## Data Collection and Preprocessing

- Text data was scraped from Wikipedia and other educational websites for each subject category.
- Data cleaning involved removing special characters, handling missing values, and tokenizing text.
- A TF-IDF vectorizer was used to convert text data into numerical features.
- The dataset was balanced to ensure equal representation of all subjects.

## Exploratory Data Analysis

- Visualizations were created to understand the distribution of categories and text lengths.
- Word clouds were generated for each subject to identify frequently occurring terms.
- Analysis of text length distribution across different subjects was performed.

## Machine Learning Models

Several classification algorithms were implemented and compared:
1. Logistic Regression
2. Random Forest Classifier
3. Support Vector Classifier (SVC)
4. XGBoost Classifier
5. GridSearchCV optimized SVC

## Results and Evaluation

- Models were evaluated using accuracy, precision, recall, F1-score, and R2 score.
- Confusion matrices were generated to visualize classification performance.
- The SVC model showed the best overall performance with an accuracy of 83.14%.

## Key Findings

1. Text Classification Performance:
   - The Support Vector Classifier (SVC) achieved the highest accuracy of 83.14%.
   - All models performed well, with accuracies ranging from 78.57% to 83.14%.

2. Subject Differentiation:
   - Some subjects like Mathematics and Physics were easier to classify accurately.
   - There was some confusion between closely related subjects (e.g., Biology and Chemistry).

3. Feature Importance:
   - Certain keywords and phrases were highly indicative of specific subjects.
   - The TF-IDF vectorization effectively captured the importance of subject-specific terms.

4. Model Comparison:
   - SVC and Logistic Regression outperformed tree-based models like Random Forest and XGBoost for this text classification task.
   - GridSearchCV optimization of SVC did not significantly improve performance over the base SVC model.

5. Data Quality:
   - The web-scraped data provided a good foundation for subject classification.
   - Balancing the dataset helped in achieving consistent performance across all subjects.

## Future Work

- Experiment with deep learning models such as LSTM or BERT for potentially improved performance.
- Expand the dataset to include more diverse and nuanced text samples.
- Implement a multi-label classification approach to handle texts that span multiple subjects.
- Develop a web application or API to allow real-time classification of user-input text.
- Explore topic modeling techniques to identify sub-topics within each subject category.

## Technologies Used

- Python
- BeautifulSoup for web scraping
- Pandas and NumPy for data manipulation
- Matplotlib and Seaborn for data visualization
- NLTK for natural language processing tasks
- Scikit-learn for machine learning models and evaluation
- XGBoost for gradient boosting
