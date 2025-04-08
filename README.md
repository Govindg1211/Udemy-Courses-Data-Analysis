# Udemy-Courses-Data-Analysis

## Introduction to Data Set

This dataset contains detailed information on Udemy courses related to Finance and
Accounting. It provides insights into various course attributes, such as course titles, pricing,
ratings, subscriber counts, and engagement metrics. The dataset also tracks whether
courses are free or paid, the number of lectures and practice tests available, and details about
their creation and publication dates. Additionally, it includes pricing information,
including both original and discounted prices, allowing for an analysis of pricing strategies
and trends. These attributes help in understanding course popularity, pricing effectiveness,
and user engagement in the Finance and Accounting domain

---

## Problem Statement: Predicting Course Popularity and Ratings on Udemy

With the rapid expansion of online education, platforms like Udemy offer thousands of
courses across diverse fields. However, not all courses achieve the same level of popularity
or receive high ratings. To enhance course performance, instructors and platform
administrators require data-driven insights to identify key factors influencing success.
Predicting a course’s subscriber count and average rating can help content creators refine
their offerings, adjust pricing strategies, and optimize marketing efforts to boost student
engagement and satisfaction.
This machine learning project aims to develop predictive models to estimate:
1. The anticipated number of subscribers for a course by analysing factors such as price,
 ratings, lecture count, publication date, and other relevant attributes.
3. The expected average rating of a course based on its features, student engagement metrics,
and pricing information.
By leveraging historical data from Udemy, our model can assist instructors in making informed
decisions about course content, pricing, and marketing, ultimately leading to a better
learning experience for students.

### Objective:

• Build machine learning models to predict:
1. Number of subscribers a course will attract.
2. Average rating a course is likely to receive.

### Key Features to Consider:

• Course Attributes: Title, number of lectures, and practice tests.
• Pricing Information: Original price and discounted price.
• Engagement Metrics: Number of reviews and wishlist status.
• Publication Details: Release date and duration on the platform.

---

 ## Machine Learning Model Implementation
 
 1. Data Splitting
    ➢ train_test_split: Divides the dataset into training and testing sets to evaluate model
performance.
    ➢ cross_val_score: Conducts cross-validation to measure model reliability and reduce
the risk of overfitting
2. Model Evaluation
   ➢ Mean Absolute Error (MAE): Calculates the average absolute difference between
actual and predicted values.
  ➢ Mean Squared Error (MSE): Emphasizes larger errors more heavily than MAE by
squaring the differences.
  ➢ R² Score: Measures how well the model explains variance in the data.
3. Hyperparameter Tuning
  ➢ Manual Hyperparameter Tuning
4. Model Selection
The code uses multiple machine learning models, mainly for regression tasks (predicting
num_subscribers and avg_rating_recent). Below is a structured breakdown of the models
used:
I. Base Models (Used in Stacking Regressor)
 These models serve as the first layer in the Stacking Regressor:
❑ Random Forest Regressor (RandomForestRegressor)
❑ Decision Tree Regressor (DecisionTreeRegressor)
❑ XGBoost Regressor (XGBRegressor)
❑ CatBoost Regressor (CatBoostRegressor)
❑ Extra Trees Regressor (ExtraTreesRegressor)
II. Meta-Model (Final Model in Stacking)
❑ MLP Regressor (MLPRegressor)
III. Stacking Regressor (StackingRegressor)
❑ A combined model integrating multiple base regressors (RandomForestRegressor,
DecisionTreeRegressor, XGBRegressor, CatBoostRegressor, and
ExtraTreesRegressor).
❑ Utilizes MLP Regressor as the final estimator (meta-model).
❑ Enhances accuracy by aggregating predictions from multiple models

---

## Techniques that have been considered

1. Data Preprocessing
➢ Utilize libraries like pandas, numpy, matplotlib, and seaborn for data
manipulation and visualization.
➢ Handle missing values by using .isnull().sum() to identify gaps and calculating
their percentage.
➢ Analyze dataset structure with .info(), summarize statistics with .describe(), and
check dimensions using .shape().

2. Exploratory Data Analysis (EDA)
➢ View the first and last few rows of the dataset using .head() and .tail().
➢ Generate a statistical summary of both numerical and categorical data using
.describe().
➢ Identify unique values within categorical columns.

3. Feature Engineering
➢ Manage missing values by calculating their percentage.
➢ Convert categorical features into numerical representations.
➢ Apply Principal Component Analysis (PCA) for dimensionality reduction while
preserving 99% of the variance

4. Feature Scaling
➢ StandardScaler: Normalizes features by subtracting the mean and scaling to unit
variance.
➢ MinMaxScaler: Rescales features to a specified range, typically between 0 and 1.

---

## Step-by-Step Project Implementation

Step 1: Import Required Libraries
• Load essential Python libraries for data manipulation, visualization, and machine learning.
Step 2: Load the Dataset
• Read the dataset into a DataFrame and check its structure to understand the data.
Step 3: Exploratory Data Analysis (EDA)
• Analyze the dataset's shape, column names, data types, and missing values.
• Visualize key variables to understand their distributions and relationships.
• Generate a correlation heatmap to identify important numerical features.
Step 4: Data Preprocessing
• Handle missing values by either filling or dropping them.
• Convert categorical variables into numerical format using encoding techniques.
Step 5: Feature Engineering
• Remove irrelevant columns that don’t contribute to prediction.
• Scale numerical features to ensure a uniform range.
Step 6: Splitting Data into Training & Testing Sets
• Divide the dataset into training and testing sets to train and evaluate models.
Step 7: - Model Selection & Training
• Train multiple regression models for prediction.
Step 8: Define Machine Learning Models
• Stacking Regressor with multiple base models.
Step 9: Train the Models
• Fit the Stacking Regressor on training data.
Step 10: Model Predictions
• Use the trained model to predict the target variables.
Step 11: Model Evaluation
• Use MAE, MSE, and R² Score to assess model performance.
Step 12: Hyperparameter Tuning
• Improve model performance by optimizing parameters using grid search.
Step 13: Model Saving & Deployment
• Save the best-performing model and use it for future predictions.

---

## Summary

This project aimed to predict the number of subscribers and course ratings for Udemy
courses using machine learning techniques. The dataset underwent comprehensive
preprocessing, including feature engineering (e.g., creating features like subscribers per
review and lecture density), handling missing values, and scaling with StandardScaler. To
enhance efficiency and model performance, Principal Component Analysis (PCA) was
applied for dimensionality reduction.
For predictive modeling, a Stacking Regressor was employed, combining multiple machine
learning algorithms such as Random Forest, XGBoost, Decision Tree, Extra Trees, and
CatBoost. A Neural Network (MLP Regressor) served as the meta-model. The models were
trained and evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and
R² Score to ensure reliable predictions.

## Conclusion

The Stacking Regressor method has shown remarkable effectiveness in forecasting both the
popularity (number of subscribers) and quality (ratings) of courses by capitalizing on the
strengths of various base models. The findings indicate that ensemble learning techniques
outperform individual models in terms of accuracy and robustness. These insights can assist
course creators, marketers, and e-learning platforms in refining their content strategies by
identifying the critical factors that contribute to course success. Potential future
enhancements could involve hyperparameter optimization, more sophisticated feature
selection, and the integration of deep learning methods to further improve predictive 
performance.
