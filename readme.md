## Zillow Regression Project

- The goal of this project is to discover the primary drivers affecting home value in three Southern California counties in the United States using a dataset from Zillow, and to create a machine learning model using linear regression that will predict home value. My initial hypothesis is that the location of the home, the number of bathrooms, and the size of the home and lot are significant features affecting home value.

- An env.py file and Codeup credentials are required to use the acquire and prepare functions for this project.
- The env.py file should contain credentials set to three variables: user, host, password

#### Data Dictionary

- bathrooms: number of bathrooms in the home, including half-baths (float64)
- bedrooms: number of bedrooms in the home (float64)
- sqft: total number of square feet of the home (float64)
- fips: numerical indicator for the county where the home is located (int)
- lot_size: the total size of the property lot (float64)
- tax_value: the taxable value of the property (float64)
- year: four-digit year the property was built (int)
- transactiondate: date of a transaction for the property in 2017 (YYYY-MM-DD format, string)
- 
- 
- 
- 

#### How to reproduce this project

- In order to reproduce this project, you will need access to the Codeup database or the .csv of the database. Acquire the database from Codeup using a SQL query, which I saved into a function in wrangle.py. The wrangle.py file has the necessary functions to acquire, prepare, and split the dataset.

- You will need to import the following python libraries into a python file or jupyter notebook: 
    - import pandas as pd
    - import numpy as np
    - import wrangle
    - from math import sqrt
    - import matplotlib.pyplot as plt
    - import seaborn as sns
    - from scipy import stats
    - from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
    - from sklearn.feature_selection import SelectKBest, f_regression 
    - from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
    - from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor

- Prepare and split the dataset. The code for these steps can be found in the wrangle.py file within this repository.
- Use pandas to explore the dataframe and scipy.stats to conduct statistical testing on the selected features.
- Use seaborn or matplotlib.pyplot to create graphs of your analyses.
- Conduct a univariate analysis on each feature using barplot for categorical variables and .hist for continuous variables.
- Conduct a bivariate analysis of each feature against tax_value and graph each finding.
- Conduct multivariate analyses of the most important features against tax_value and graph the results.
- Create models (OLS regression, LassoLars, TweedieRegressor) with the most important selected features using sklearn.
- Train each model and evaluate its accuracy on both the train and validate sets.
- Select the best performing model and use it on the test set.
- Graph the results of the test using probabilities.
- Document each step of the process and your findings.

#### Key Findings, Recommendations, and Takeaways