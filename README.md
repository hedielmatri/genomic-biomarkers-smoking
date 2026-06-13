# Genomic Predisposition to Smoking

## Current Status: Active Research
This project is currently ongoing. Final biological insights will be updated upon project completion.

## About the Project
This project finds genetic links to smoking. I built this during my research internship at the Institut Pasteur (IBMS Lab). The goal is to predict if a person is a smoker based on their genes. The dataset contains over 200,000 genetic features.

## Data Privacy Notice
I cannot share the raw genomic data. This is due to strict patient privacy rules. However, I have provided my full code pipeline. You can read the code to see exactly how I processed the data and trained the models.

## Tools and Methods
I used several tools to handle the big data and train the models:
* **Data Processing:** I used Polars to load and clean the massive dataset quickly.
* **Feature Selection:** I filtered the genes using Minor Allele Frequency (MAF). I also used Chi-squared tests and LASSO regression to find the most important genes.
* **Machine Learning:** I trained predictive models using XGBoost and Support Vector Machines (SVM).
* **Optimization:** I used Optuna to find the best settings for the models.
* **Evaluation:** I tested the models using ROC-AUC scores and Youden's J-statistic.

## Repository Structure
* `notebooks/data_preparation.ipynb`: This file shows how I cleaned the data and selected the features.
* `notebooks/data_inference.ipynb`: This file shows how I trained the SVM and XGBoost models.
* `requirements.txt`: This lists the tools needed to run the code.
