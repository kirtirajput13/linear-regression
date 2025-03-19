# Linear Regression with California Housing Dataset

## Overview
This project demonstrates the implementation of a **Linear Regression** model to predict median house values using the **California Housing Dataset**. The goal is to showcase the fundamental steps involved in a typical data science workflow, including data exploration, preprocessing, model training, evaluation, and visualization.

## Dataset
The dataset used in this project is the **California Housing Dataset**, which contains information about housing prices in California. It includes features such as median income, house age, average number of rooms, and more. The target variable is the median house value for each district.

### Features
- **MedInc**: Median income in the block group.
- **HouseAge**: Median house age in the block group.
- **AveRooms**: Average number of rooms per household.
- **AveBedrms**: Average number of bedrooms per household.
- **Population**: Total population in the block group.
- **AveOccup**: Average number of household members.
- **Latitude**: Latitude of the block group.
- **Longitude**: Longitude of the block group.

### Target
- **MedHouseVal**: Median house value for households in the block group.

## Steps in the Project

### 1. Data Loading and Exploration
- The dataset is loaded using `fetch_california_housing` from the `sklearn.datasets` module.
- Basic exploration includes displaying the first few rows, checking for missing values, and summarizing the dataset statistics.

### 2. Data Visualization
- A **correlation matrix** is plotted to understand the relationships between features and the target variable.
- This helps identify which features are most strongly related to the median house value.

### 3. Data Preprocessing
- The dataset is split into **training** and **testing** sets using an 80-20 split.
- Features are standardized using `StandardScaler` to ensure all features are on the same scale, which is important for linear regression.

### 4. Model Training
- A **Linear Regression** model is initialized and trained on the training data.
- The model learns the relationship between the features and the target variable.

### 5. Model Evaluation
- Predictions are made on the test set, and the model's performance is evaluated using:
  - **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values.
  - **Root Mean Squared Error (RMSE)**: The square root of MSE, providing a more interpretable error metric.
  - **R-squared (R²)**: Indicates the proportion of variance in the target variable that is predictable from the features.

### 6. Visualization of Results
- **Actual vs Predicted Prices**: A scatter plot is created to compare the actual and predicted house values. A red line represents the ideal case where predictions match actual values.
- **Residual Plot**: A scatter plot of residuals (differences between actual and predicted values) is used to check for homoscedasticity (constant variance of residuals).

### 7. Feature Importance
- The coefficients of the linear regression model are analyzed to understand the importance of each feature in predicting the target variable.

## Results
The model's performance is evaluated using the following metrics:
- **Mean Squared Error (MSE)**: [Value]
- **Root Mean Squared Error (RMSE)**: [Value]
- **R-squared (R²)**: [Value]

The visualizations provide insights into the model's predictions and the distribution of residuals.

## How to Run the Code
1. Open the provided Jupyter Notebook or Google Colab file.
2. Run each cell sequentially to load the dataset, preprocess the data, train the model, and evaluate its performance.
3. The visualizations and evaluation metrics will be displayed as outputs.

## Dependencies
To run this project, you need the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install these libraries using `pip`:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Conclusion
This project provides a comprehensive example of how to build and evaluate a linear regression model using the California Housing Dataset. It covers all the essential steps in a data science workflow, making it a great addition to your portfolio. The code is well-commented and easy to follow, making it suitable for both beginners and experienced data scientists.

## License
This project is open-source and available under the MIT License. Feel free to use, modify, and distribute it as needed.

---

This README file provides a clear and detailed explanation of the project, making it easy for anyone to understand and replicate the work.
