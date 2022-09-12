## Zillow Regression Project

- The goal of this project is to discover the main features affecting property tax assessed value for single family residences in three Southern California counties and to use these features to develop a machine learning model that will accurately predict future home values for the most commonly sold homes. These predictions can provide customers with realistic estimates of tax value when they use our website. My initial hypothesis is that the location of the home, the number of bathrooms, and the size of the home and lot are significant features affecting home value.

- Initial questions:
    - What is the relationship between square feet and tax value? Does the size of the home have a large impact on the overall tax assessment?
    - Which county has the most valuable properties? Can that value be determined by the year the home was built?
    - Does the number of bedrooms have an effect on tax value? Number of bathrooms? Total number of bedrooms and bathrooms?

- An env.py file and Codeup credentials are required to use the acquire and prepare functions for this project.
- The env.py file should contain credentials set to three variables: user, host, password

#### Project Planning
- Acquire the dataset from the Codeup database using SQL
- Prepare the data with the intent to discover the main drivers of assessed tax value; clean the data and encode categorical features if necessary; ensure that the data is tidy
- Split the data into train, validate, and test datasets using a 60/20/20 split and a random seed of 217
- Explore the data:
    - Univariate, bivariate, and multivariate analyses; statistical tests for significance, find the three primary features affecting tax value, use sqft, bedrooms, and bathrooms for the first model
    - Create graphical representations of the analyses
    - Ask more questions about the data
    - Document findings
- Train and test models:
    - Establish a baseline using the mean tax value
    - Select key features and train multiple regression models
    - Test the model on the validate set, adjust for overfitting if necessary
- Select the best model for the project goals:
    - Determine which model performs best on the validate set
- Test and evaluate the model:
    - Use the model on the test set and evaluate its performance (RMSE, r2 score, etc.)
    - Visualize the data using an array of probabilities on the test set
- Document key findings and takeaways, answer the questions
- Create a final report

#### Data Dictionary

- bathrooms: number of bathrooms in the home, including half-baths (float64)
- bedrooms: number of bedrooms in the home (float64)
- sqft: total number of square feet of the home (float64)
- fips: numerical indicator for the county where the home is located (int)
- lot_size: the total size of the property lot (float64)
- tax_value: the taxable value of the property (float64)
- year: four-digit year the property was built (int)
- transactiondate: date of a transaction for the property in 2017 (YYYY-MM-DD format, string)
- bed_bath: the total number of bedrooms and bathrooms in the home (float64)
- lot_minus_home: the difference in square feet between the overall lot size and the home size (float64)
- zipcode: the zipcode of the property (float64)
- fips_6037: code for Los Angeles County (uint8)
- fips_6059: code for Orange County (uint8)
- fips_6111: code for Ventura County (uint8)

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

- After running four models on my train and validate sets, I decided to use the polynomial linear regression model because it provided the lowest RMSE and highest r2 score overall.

- I used the eight most significant features for assessed tax value (bathrooms, square feet of the home, Los Angeles County, total number of bedrooms and bathrooms, lot size, year built, the difference in lot size and home size, and the zip code). I selected a degree multiplier of 2. The RMSE of the selected model was 133682 on train, 135575 on validate, and 134578 on test. The test r2 score was .20.

- Takeaways: the biggest drivers of tax value are the number of bathrooms, the size of the home in square feet, and the number of bedrooms. The addition of zip code, Los Angeles County, total bedrooms and bathrooms, and lot size excluding home square footage decreased the root mean squared error and raised the explained variance score. The models all performed above the baseline RMSE. 

- The selected model has a lower root mean squared error than baseline predictions, but can only account for 20% of the variance in home values. Bathrooms are the most significant single feature that affects home value, but there are many other factors to consider in order to get a better prediction.

- I recommend obtaining accurate data on the number of stories the home has, as well as parking structures or spaces in order to more accurately predict home value. I also recommend adding crime rates and school ratings to the dataset to see if it has any effect on the model's performance. We could also use the type of single family residence (house, condo, townhome, etc.) in order to tune the model. We can also investigate how much the tax assessed value increased annually over the last 50 years in order to make better predictions.

- If I had more time, I would do more feature engineering on the zip codes to see if there is a relationship between that and home value, home size, and home age. I would also test non-linear regression models to see if they perform better on the data we currently have.