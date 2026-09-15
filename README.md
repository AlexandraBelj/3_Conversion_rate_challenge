@"
# Conversion Rate Challenge

## Overview

This project is part of the Jedha Data Science certification.

The objective is to predict whether a website visitor will subscribe to the Data Science Weekly newsletter.

This is a supervised machine learning binary classification problem. The primary evaluation metric is the F1-score.

## Project Structure

- ``data/raw/`` — competition datasets (not tracked by Git)
- ``notebooks/`` — analysis and modeling notebooks
- ``outputs/figures/`` — exported visualizations
- ``outputs/metrics/`` — model evaluation results
- ``outputs/submissions/`` — competition submission files

## Notebooks

1. ``01_EDA_Preprocessing.ipynb`` — exploratory data analysis and preprocessing
2. ``02_Modeling_Optimization.ipynb`` — baseline, cross-validation and model optimization
3. ``03_Final_Model_Submission_Insights.ipynb`` — final model, competition submission and business insights

## Data

Place the competition datasets in ``data/raw/``:

- ``conversion_data_train.csv``
- ``conversion_data_test.csv``

The training dataset contains the target variable and is used for model development and internal evaluation.

The competition test dataset contains no target labels and is reserved for generating the final competition submission.
"@ | Set-Content README.md