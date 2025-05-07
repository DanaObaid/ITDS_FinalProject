## ITDS Final Project - Data Science Lab

## Overview
This project is the end-term assignment for the **Data Science Lab**, focusing on applying regression and time series forecasting techniques learned during the course. It is divided into three main exercises:
1. **Univariate Regression on Analytical Functions**: Building and evaluating regression models for three analytical functions (f1, f2, f3) and a univariate synthetic dataset generated using scikit-learn.
2. **Multivariate Regression on Synthetic Data**: Modeling a synthetic dataset with multiple features to explore multivariate regression techniques.
3. **Temperature Series Forecasting**: Forecasting daily mean temperatures for Honolulu, Hawaii, using a historical weather dataset, analyzing trends, seasonality, and prediction accuracy.

The project is implemented in a Jupyter Notebook (`ITDS_FinalProject.ipynb`) using Python with libraries such as NumPy, Pandas, Matplotlib, and scikit-learn. The notebook contains the code, visualizations, and detailed analysis for all tasks, demonstrating skills in data preprocessing, model selection, evaluation, and visualization.

## Author
- **Name**: Dana Obid
- **Neptun Code**: GPFH4Y

## Requirements
To run this project, ensure you have the following dependencies installed:
- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- scikit-learn

Install the required libraries using pip:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Project Structure
The project is submitted as a ZIP file via Canvas, containing:
- **ITDS_FinalProject.ipynb**: The main Jupyter Notebook with all code, visualizations, and analysis.
- **task_1_functions.png**: A generated plot visualizing the analytical functions with train/test data splits.
- **task_2_pred_vs_actual.png**: A generated plot for multivariate regression, showing predicted vs. actual values.
- **README.md**: This file, providing an overview and instructions.

## Setup and Execution
1. **Clone or Unzip**:
   - Clone the GitHub repository (see link below) or unzip the submitted ZIP file.
2. **Set Up Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```
4. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
5. **Run the Notebook**:
   - Open `ITDS_FinalProject.ipynb`.
   - Ensure any required datasets (e.g., Honolulu temperature data) are available in the correct path or generated within the notebook.
   - Run all cells sequentially to execute the code, generate visualizations, and view results.
6. **Alternative**: Upload the notebook to **Google Colab** for execution without local setup.

## Project Details

### Exercise 1: Univariate Regression on Analytical Functions
- **Objective**: Model three analytical functions and a univariate synthetic dataset using regression techniques.
- **Functions**:
  - f1(x) = x * sin(x) + 2x
  - f2(x) = 10 * sin(x) + x²
  - f3(x) = sign(x) * (x² + 300) + 20 * sin(x)
- **Synthetic Dataset**: Generated via `make_regression` with 100 samples, 1 feature, and noise (standard deviation = 10).
- **Models**: Linear Regression, Ridge Regression, Random Forest Regressor, Support Vector Regression (SVR).
- **Key Steps**:
  - Generated 100 samples over [-20, 20], split into 70% training and 30% testing sets.
  - Evaluated models using R² and Mean Squared Error (MSE) metrics.
  - Visualized functions and train/test data in `task_1_functions.png`.
- **Findings**:
  - Random Forest and SVR outperformed linear models due to non-linear patterns in f1, f2, and f3.
  - Linear models struggled with f3’s discontinuity caused by the sign(x) term.
  - The synthetic dataset was effectively modeled by all methods, with Random Forest achieving the highest R² due to its robustness to noise.

### Exercise 2: Multivariate Regression on Synthetic Data
- **Objective**: Model a synthetic dataset with multiple features to explore multivariate regression techniques.
- **Dataset**:
  - Generated using scikit-learn’s `make_regression` with:
    - 100 samples.
    - 5 features (`n_features=5`).
    - 3 informative features (`n_informative=3`).
    - Noise (standard deviation = 10).
    - Random seed (`random_state=42`) for reproducibility.
  - The target variable is a linear combination of the 3 informative features, with noise to simulate real-world complexity.
- **Models**: Linear Regression, Ridge Regression, Random Forest Regressor, Support Vector Regression (SVR).
- **Key Steps**:
  - Split the dataset into 70% training and 30% testing sets using `train_test_split` with `random_state=42`.
  - Applied feature scaling using `StandardScaler` to normalize features, improving performance for models like SVR.
  - Trained models on the training set and predicted on the test set.
  - Evaluated models using R² and MSE metrics.
  - Visualized results with:
    - A scatter plot of predicted vs. actual values (`task_2_pred_vs_actual.png`).
    - A bar plot of feature importance for Random Forest to highlight the contribution of each feature.
- **Findings**:
  - **Linear Regression and Ridge Regression** performed best due to the linear nature of the dataset, with Ridge slightly outperforming in cases of feature correlation.
  - **Random Forest** effectively identified the 3 informative features, as shown in feature importance analysis, but slightly underperformed linear models for this linear relationship.
  - **SVR** with a linear kernel yielded results comparable to linear models, while the RBF kernel overfit the noise, reducing performance.
  - The 2 non-informative features tested the models’ ability to ignore irrelevant inputs, with Ridge and Random Forest handling this well.
  - Noise limited the maximum achievable R², as some variance was irreducible.
- **Challenges**:
  - Handling noise and non-informative features required robust models.
  - Visualizing high-dimensional data was addressed through predicted vs. actual plots and feature importance analysis.
- **Visualizations**:
  - Predicted vs. actual scatter plot to assess model fit.
  - Feature importance bar plot for Random Forest to highlight the contribution of each feature.

### Exercise 3: Temperature Series Forecasting
- **Objective**: Forecast daily mean temperatures for Honolulu and analyze trend and seasonality.
- **Dataset**: Historical daily mean temperature data for Honolulu, characterized by stable temperatures (20–25°C) and weak seasonality.
- **Model**: Random Forest Regressor with tuned hyperparameters (`n_estimators=200`, `max_depth=10`, `random_state=42`).
- **Key Steps**:
  - Applied a rolling window (size=10) to create lagged features, transforming the time series into a supervised learning problem.
  - Split data into training and testing sets.
  - Evaluated using R² (0.6625) and MSE (0.7781).
  - Analyzed series similarity, trend capture, and seasonality prediction.
- **Findings**:
  - The training and testing series were moderately similar, as indicated by the R² score.
  - The stable temperature trend (20–25°C) was well-captured, but the weak seasonality was not fully predicted due to subtle patterns and reliance on lagged features.
  - Multi-step forecasting beyond one day was challenging due to error accumulation and lack of recursive forecasting or explicit seasonal features.
- **Experimentation**:
  - Tuned the window size and model hyperparameters to improve performance.
  - The reported R² and MSE reflect the tuned model’s results.

## GitHub Repository
The project is hosted on GitHub at:  
[https://github.com/DanaObaid/ITDS_FinalProject](https://github.com/DanaObaid/ITDS_FinalProject)  

## Notes for Reviewers
- The notebook is self-contained, with all code, explanations, and results included.
- If the Honolulu temperature dataset is required but not included, please contact me for clarification or provision.
- For any issues running the notebook, reach out via Canvas or email.

## Contact
- **Name**: Dana Obid
- **Neptun Code**: GPFH4Y
