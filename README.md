# Genomic Predisposition to Smoking

## About the Project
This project finds genetic links to smoking. This was built during a research internship at the Institut Pasteur (IBMS Lab) under founder Dr. Alia Benkhala. The goal is to predict if a person is a smoker based on their genes. The dataset contains about 250,000 genetic features. Cross ancestry enrichment analyses comparing Caucasian and African cohorts were applied to isolate conserved genetic architectures and identify overlapping genes.


## Biological Implications
The data suggests a few insights:
- Out of over 250,000 variants, fewer than 700 consistently survived statistical filtering. This suggests the true genetic driver for smoking might be incredibly sparse and easily drowned out by random genetic noise.  
- In the EpistaticFusionNet architecture, combining a direct additive linear shortcut with a nonlinear branch outperformed purely nonlinear or purely linear models. This suggests that genomic risk operates by a dual mechanism where baseline additive risk and complex interaction networks coexist.
- Population structure eigenvectors extracted via SVD required dozens of components to resolve background variation. This highlights specific genetic drift of a population can easily masquerade as smoking risk loci if ancestry covariates are not strictly decoupled.  

## Data Privacy
The raw genomic data cannot be shared. However, the full code pipeline is provided.

## Tools and Methods
 Tools were used to handle the data and train the models:
- Data processing:
    - Polars was used to load and clean the dataset quickly.
    - A PLINK QC pipeline was engineered for quality control. This included autosomal filtering, variant missingness thresholds, Minor Allele Frequency (MAF) filtering, and linkage disequilibrium (LD) pruning.
    - Principal Component Analysis (PCA) and Singular Value Decomposition (SVD) were used to try and control for genetic stratification.
- Feature selection: Genes were filtered using MAF, Chi-squared tests, ANOVA F-tests, Recursive Feature Elimination (RFE), and LASSO regression to try and find the most important variants. A custom permutation method was also built to test features against random noise.
- Machine learning: Predictive models were trained using XGBoost, Support Vector Machines (SVM), Random Forest, and Logistic Regression. A Polygenic Risk Score (PRS) model was tested using Ridge Regression. To try and model non-linear genomic risk, a Feedforward Neural Network (EpistaticFusionNet) was architected using Batch Norm1D regularization and SiLU activations. Double ML was also applied for causal AI.
- Optimisation: Optuna's Bayesian framework was used to find the best settings for the models.
- Evaluation: The models were evaluated using ROC-AUC scores, Precision Recall AUC, Youden's J-statistic, and classification reports. Models were also benchmarked comparing Caucasian and African cohorts.


## Repository Structure
- notebooks/data_preparation.ipynb: This file shows how the data was cleaned and formatted.
- notebooks/data_inference.ipynb: This file shows how the baseline models were trained, how stability selection was run, and how Optuna was used for optimisation.
- inference_by_PLINK_preparation.ipynb: This script contains the feature selection, Optuna optimisation, and SVM/Random Forest models followed by a plink preparation.
- PLINK_filtering.ipynb: This script executes the PLINK QC pipeline, LD pruning, and PCA generation.
- AI_solutions_testing_for_result_comparaison.ipynb: This file tests the Polygenic Risk Score modeling and validates models comparing Caucasian and African cohorts.
- rfe_testing.ipynb: This script executes the PLINK logistic hybrid regression and tests 10-fold CV with RFE.
- data_nn_snp_association.py: The PyTorch script building the EpistaticFusionNet, SVD covariate extraction, and permutation feature selection.
