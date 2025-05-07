
## ITDS Final Project - Data Science Lab

## Overview
This project is the end-term assignment for the **Data Science Lab**, focusing on applying regression and time series forecasting techniques learned during the course. It is divided into two main exercises:
1. **Univariate Regression on Analytical Functions**: Building and evaluating regression models for three analytical functions (f1, f2, f3) and a synthetic dataset generated using scikit-learn.
2. **Temperature Series Forecasting**: Forecasting daily mean temperatures for Honolulu, Hawaii, using a historical weather dataset, analyzing trends, seasonality, and prediction accuracy.

The project is implemented in a Jupyter Notebook (`ITDS_FinalProject.ipynb`) using Python with libraries such as NumPy, Pandas, Matplotlib, and scikit-learn. The notebook contains the code, visualizations, and detailed analysis for both tasks, demonstrating skills in data preprocessing, model selection, evaluation, and visualization.

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
- **task_2_1_functions.png**: A generated plot visualizing the analytical functions with train/test data splits.
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
- **Objective**: Model three analytical functions and a synthetic dataset using regression techniques.
- **Functions**:
  - f1(x) = x * sin(x) + 2x
  - f2(x) = 10 * sin(x) + x²
  - f3(x) = sign(x) * (x² + 300) + 20 * sin(x)
  - Synthetic dataset generated via `make_regression` with noise.
- **Models**: Linear Regression, Ridge Regression, Random Forest Regressor, Support Vector Regression (SVR).
- **Key Steps**:
  - Generated 100 samples over [-20, 20], split into 70% training and 30% testing sets.
  - Evaluated models using R² and MSE metrics.
  - Visualized functions and train/test data in `task_2_1_functions.png`.
- **Findings**:
  - Random Forest and SVR outperformed linear models due to non-linear patterns in f1, f2, and f3.
  - Linear models struggled with f3’s discontinuity (sign(x) term).
  - Synthetic dataset was effectively modeled, with Random Forest achieving the highest R².

### Exercise 2: Temperature Series Forecasting
- **Objective**: Forecast daily mean temperatures for Honolulu and analyze trend and seasonality.
- **Dataset**: Honolulu temperature data, characterized by stable temperatures (20–25°C) and weak seasonality.
- **Model**: Random Forest Regressor with tuned hyperparameters (`n_estimators=200`, `max_depth=10`).
- **Key Steps**:
  - Applied a rolling window (size=10) to create lagged features.
  - Split data into training and testing sets.
  - Evaluated using R² (0.6625) and MSE (0.7781).
  - Analyzed series similarity, trend, and seasonality capture.
- **Findings**:
  - The training and testing series are moderately similar.
  - The stable trend was well-captured, but weak seasonality was not fully predicted.
  - Multi-step forecasting (>1 day) is limited due to error accumulation and lack of recursive forecasting or seasonal features.

## GitHub Repository
The project is hosted on GitHub at:  
[[https://github.com/DanaObaid/ITDS_FinalProject](https://github.com/DanaObaid/ITDS_FinalProject)](https://github.com/your-username/ITDS_FinalProject)  
[Note: Replace with the actual GitHub URL after uploading your project. Ensure the repository is public or accessible to reviewers.]

## Notes for Reviewers
- The notebook is self-contained, with all code, explanations, and results included.
- If the Honolulu temperature dataset is required but not included, please contact me for clarification or provision.
- For any issues running the notebook, reach out via Canvas or email.

## Contact
- **Name**: Dana Obid
- **Neptun Code**: GPFH4Y
