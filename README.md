# Comparative Analysis of Machine Learning Architectures for Customer Churn Prediction and Behavioral Segmentation in Banking

**CM763 AI Course Project**

**Authors:** 
- **Nan**	1137130
- **Adam** 1137166
- **Fon** 1137183

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_UZYci-gJIEgGiohh6NDDNa4x_ErSvOm?usp=sharing)

## Project Overview
This study conducts a comprehensive analysis of six machine learning architectures for churn prediction and behavioral segmentation. By integrating **XGBoost** for prediction and **Standard K-Means** for descriptive analytics, the framework provides actionable insights to mitigate customer attrition and optimize marketing resources.

## Methodology

### Stage 1: Churn Prediction (Model Benchmarking)
We rigorously evaluated 6 models using a standardized preprocessing pipeline and **5-Fold Cross-Validation**.

| Model Name | Mean Train Accuracy | Mean Validation Accuracy | Standard Deviation |
| :--- | :---: | :---: | :---: |
| **XGBoost (Champion)** | **89.04%** | **86.33%** | **0.25%** |
| LightGBM | 91.31% | 86.11% | 0.17% |
| Random Forest | 100.00% | 86.02% | 0.46% |
| CCP-Net (Adapted) | 86.29% | 85.94% | 0.37% |
| Custom MLP | 87.18% | 85.92% | 0.29% |
| Logistic Regression | 81.13% | 81.04% | 0.53% |


**Key Finding:** XGBoost outperformed deep learning models (CCP-Net) in absolute accuracy and stability for this structured tabular dataset.

### Stage 2: Behavioral Segmentation
Following the prediction phase, 8,717 "Not Churn" customers were segmented using **Standard K-Means** ($K=4$), which was chosen over Autoencoder and DBSCAN for its superior business interpretability.


## Customer Personas (The 4 Segments)
| Cluster | Persona | Key Characteristics | Strategy |
| :--- | :--- | :--- | :--- |
| **1** | **Established VIPs** | Balance ≈ $120k, 100% Active | High-touch retention |
| **3** | **The Dormant Rich** | Balance ≈ $121k, 0% Activity | Urgent re-activation |
| **2** | **The Transactors** | Low balance ($762), High usage | Deposit growth |
| **0** | **The No-Card Savers** | Healthy balance, No Credit Card | Cross-selling cards |

## Tech Stack
- **Language:** Python
- **Libraries:** XGBoost, Scikit-learn, Pandas, Matplotlib, Seaborn 
- **Dataset:** Bank Customer Churn Modelling (10,000 records)

## Environment Setup & Requirements
To run this project, you need **Python 3.8+** and the following libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `seaborn`

## Installation & Environment Setup

To ensure the model runs correctly and reproduces the **86.33% Mean Validation Accuracy** reported in the study, please follow these setup instructions.

### 1. Prerequisites
- **Python 3.8+**
- **Jupyter Notebook** or **Google Colab**

### 2. Required Libraries
Install the necessary dependencies using pip:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm torch matplotlib seaborn
```
## Execution Instructions
Follow these steps to reproduce the results:

### Step 1: Repository Setup

Clone this repository to your local machine:

```bash
git clone https://github.com/s1137183/AI-Customer-Retention-Banking.git
cd AI-Customer-Retention-Banking
```

### Step 2: Data Configuration

1. Create a folder named `data` in the root directory.

2. Place the dataset file (`Churn_Modelling.csv`) inside this folder.

3. Note for Colab Users: The notebook currently looks for `/content/data/Churn_Modelling.csv.` If running locally, please update the path in Section 1 to `data/Churn_Modelling.csv`.
   

### Step 3: Running the Analysis
Open `Customer_Retention_AI.ipynb` and execute the cells in the following order:

1. **Setup & Preprocessing (Sections 0-1):**
   - Import libraries and load the dataset.
   - Define and fit the `ColumnTransformer` (StandardScaler + OneHotEncoder).

2. **Model Benchmarking (Sections 3-4):**
   - Execute **Section 3** to run the **5-Fold Cross-Validation** loop for all 6 models.
   - Execute **Section 4** to visualize the comparison results (Box Plots & Overfitting Graphs).

3. **Final Model Evaluation (Section 5):**
   - Train the final **XGBoost** model (Winner) on an 80/20 split.
   - Generate the **Confusion Matrix**, **Learning Curves**, and **Feature Importance** plot.

4. **Behavioral Segmentation (Section 6):**
   - Run **Section 6 (Part 2)** to perform Standard K-Means Clustering.
   - Verify the **Elbow Method** plot (Optimal $K=4$) and view the Cluster Analysis table.
   - *(Optional)* Execute **Section 6 (v2.0 & v2.1)** to compare results with Autoencoder-based clustering.


### Step 4: Verification

- Confirm that the XGBoost Validation Accuracy is approximately 86.33%.

- Verify the Elbow Method plot shows the optimal clusters at K=4.
---

## References & Data Sources
- **CCP-Net Model:** Adapted from *Customer churn prediction model based on hybrid neural networks* [Nature, 2024]. [View Source](https://www.nature.com/articles/s41598-024-79603-9)
- **Dataset:** *Customer Churn Prediction using Artificial Neural Network* via Vishal815. [GitHub Repository](https://github.com/vishal815/Customer-Churn-Prediction-using-Artificial-Neural-Network)

