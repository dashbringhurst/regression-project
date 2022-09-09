import pandas as pd
import numpy as np
import os
import wrangle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from math import sqrt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.feature_selection import SelectKBest, f_regression 
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import warnings
warnings.filterwarnings("ignore")

# create a function to print the regression errors for the model
def regression_errors(y, yhat):
    '''This function takes in two arguments, a previously assigned target variable (y) and the model predictions 
    (yhat). The function calculates the sum of squares error, explained sum of squares, total sum of squares,
    mean squared error, and root mean squared error. It prints strings for each value to the first decimal.'''
    # calculate the sum of squares error from the selected variables
    SSE = mean_squared_error(y, yhat)*len(y)
    # calculate the explained sum of squares from the predictions and the baseline
    ESS = sum((yhat - y.mean())**2)
    # calculate the total sum of squares
    TSS = ESS + SSE
    # calculate the mean squared error
    MSE = mean_squared_error(y, yhat)
    # calculate the root mean squared error
    RMSE = sqrt(mean_squared_error(y, yhat))
    # print the calculated values for each
    print(f'Model SSE is: {"{:.1f}".format(SSE)}')
    print(f'Model ESS is: {"{:.1f}".format(ESS)}')
    print(f'Model TSS is: {"{:.1f}".format(TSS)}')
    print(f'Model MSE is: {"{:.1f}".format(MSE)}')
    print(f'Model RMSE is: {"{:.1f}".format(RMSE)}')

# create a function to print the baseline mean errors
def baseline_mean_errors(y, baseline):
    '''This function takes in a single argument, y (the target variable) and prints the sum of squares
    error, mean squared error, and root mean squared error for baseline.'''
    # calculate the baseline sum of squares error
    SSE_baseline = mean_squared_error(y, baseline)*len(y)
    # calculate the baseline mean squared error
    MSE_baseline = mean_squared_error(y, baseline)
    # calculate the baseline root mean squared error
    RMSE_baseline = sqrt(mean_squared_error(y, baseline))
    # print the calculated values for each baseline error
    print(f'SSE baseline: {"{:.1f}".format(SSE_baseline)}')
    print(f'MSE baseline: {"{:.1f}".format(MSE_baseline)}')
    print(f'RMSE baseline: {"{:.1f}".format(RMSE_baseline)}')

# create a function to determine if the model performs better than baseline
def better_than_baseline(y, yhat, baseline):
    '''This function takes in two arguments, y (target variable) and yhat (model predictions) and calculates the 
    model SSE, MSE, and RMSE against the baseline. The function prints three strings, one for each result, with a
    boolean for whether or not the model value is better than baseline value.'''
    SSE = mean_squared_error(y, yhat)*len(y)
    SSE_baseline = mean_squared_error(y, baseline)*len(y)
    MSE = mean_squared_error(y, yhat)
    MSE_baseline = mean_squared_error(y, baseline)
    RMSE = sqrt(mean_squared_error(y, yhat))
    RMSE_baseline = sqrt(mean_squared_error(y, baseline))
    
    print(f'Model SSE is better than SSE baseline: {SSE < SSE_baseline}')
    print(f'Model MSE is better than MSE baseline: {MSE < MSE_baseline}')
    print(f'Model RMSE is better than RMSE baseline: {RMSE < RMSE_baseline}')

def plot_residuals(y, yhat):
    '''This function takes in two arguments, y (target variable) and yhat (model predictions) and returns a 
    scatterplot of the residuals of the target variable.'''
    residuals = y - yhat
    plt.scatterplot(x=y, y=residuals)
    plt.xlabel('Home Value')
    plt.ylabel('Residuals')
    plt.title('Residuals for Home Value')
    plt.show()

def lasso_lars_model(X_train_scaled, X_validate_scaled, y_train, y_validate, train, a):
    # create the model object
    lars = LassoLars(alpha=a)
    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lars.fit(X_train_scaled, y_train.tax_value)

    # predict train
    y_train['tax_pred_lars'] = lars.predict(X_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_pred_lars)**(1/2)

    # predict validate
    y_validate['tax_pred_lars'] = lars.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_pred_lars)**(1/2)

    train['yhat'] = lars.predict(X_train_scaled)

    r2 = r2 = r2_score(train.tax_value, train.yhat)

    RMSE = sqrt(mean_squared_error(train.tax_value, train.yhat))
    RMSE_baseline = sqrt(mean_squared_error(train.tax_value, train.baseline))
    better =  RMSE < RMSE_baseline

    return ['Lasso Lars', rmse_train, rmse_validate, r2, better]

def glm_model(X_train_scaled, X_validate_scaled, y_train, y_validate, train, p, a):
    # create the model object
    glm = TweedieRegressor(power=p, alpha=a)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train_scaled, y_train.tax_value)

    # predict train
    y_train['value_pred_glm'] = glm.predict(X_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.value_pred_glm)**(1/2)

    # predict validate
    y_validate['value_pred_glm'] = glm.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.value_pred_glm)**(1/2)

    train['yhat'] = glm.predict(X_train_scaled)
    
    r2 = r2_score(train.tax_value, train.yhat)
    
    RMSE = sqrt(mean_squared_error(train.tax_value, train.yhat))
    RMSE_baseline = sqrt(mean_squared_error(train.tax_value, train.baseline))
    better =  RMSE < RMSE_baseline

    return ['Tweedie Regressor', rmse_train, rmse_validate, r2, better]


def poly_lm(X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, train, d):
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=d)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train_scaled)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate_scaled)
    X_test_degree2 = pf.transform(X_test_scaled)
    
    # create the model object
    lm2 = LinearRegression()

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.tax_value)

    # predict train
    y_train['value_pred_lm2'] = lm2.predict(X_train_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.value_pred_lm2)**(1/2)

    # predict validate
    y_validate['value_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.value_pred_lm2)**(1/2)

    train['yhat'] = lm2.predict(X_train_degree2)

    r2 = r2_score(train.tax_value, train.yhat)
    
    RMSE = sqrt(mean_squared_error(train.tax_value, train.yhat))
    RMSE_baseline = sqrt(mean_squared_error(train.tax_value, train.baseline))
    better =  RMSE < RMSE_baseline

    return ["Poly Linear Regression", rmse_train, rmse_validate, r2, better]

def lrm(X_train_scaled, X_validate_scaled, y_train, y_validate, train):
    
    # create the model object
    lm = LinearRegression()

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train_scaled, y_train.tax_value)

    # predict train
    y_train['value_pred_lm'] = lm.predict(X_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.value_pred_lm)**(1/2)

    # predict validate
    y_validate['value_pred_lm'] = lm.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.value_pred_lm)**(1/2)

    train['yhat'] = lm.predict(X_train_scaled)

    r2 = r2_score(train.tax_value, train.yhat)
    
    RMSE = sqrt(mean_squared_error(train.tax_value, train.yhat))
    RMSE_baseline = sqrt(mean_squared_error(train.tax_value, train.baseline))
    better =  RMSE < RMSE_baseline

    return ["OLS", rmse_train, rmse_validate, r2, better]

def model_performance(m1, m2, m3, m4):
    df = pd.DataFrame([m1,m2,m3,m4])
    df = df.rename(columns={0:'Model', 1:'Train RMSE', 2:'Validate RMSE', 3:'r2 score', 4:'Better than Baseline'})
    return df