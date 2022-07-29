"""A library of standard procedures used in regression analysis.

This module requires that `numpy` and `pandas` be installed within the Python
environment you are exporting this module to.

Functions
---------
proj_mat(x)
    Projection matrix of an array.
lse(x, y)
    Least squares estimates. 
weighted_lse(x, y, w)
    Weighted least squares estimates of regression coefficients.
weighted_pred(x, y, w)
    Estimates of response obtained using weighted least squares.
weighted_res(x, y, w)
    Residuals obtained using weighted least squares.
res(x, y)
    Residuals for least squares estimates.   
del_res(x, y)
    Deleted residuals for the least squares estimates.  
sse(x, y)
    Residual sum of squares.
mse(x, y)
    Mean squared error.
ssto(y)
    Total sum of squares of an array.
coeff_det(x, y)
    Coefficient of multiple determination.
adj_coeff_det(x, y)
    Adjusted coefficient of multiple determination.
Cp(x_full, x_red, y)
    Mallow's C_p criterion.
press(x, y)
    Prediction sum of squares.
aic(x, y)
    Akaike's information criterion.
sbc(x, y)
    Schwarz' Bayesian criterion.
"""

import numpy as np
import pandas as pd

def proj_mat(x):
    """Projection matrix.
    
    The projection matrix can be matrix multiplied by a data array to give the
    projection of the data array onto the column space of the input array.
    
    Parameters
    ----------
    x : array_like
        Input array, cannot be singular?
        
    Returns
    -------
    r : ndarray
        Projection matrix onto the column space of the input array
    """
    
    inv = np.linalg.inv(np.matmul(x.T, x))
    r = np.matmul(np.matmul(x, inv), x.T)
    return r

def fit(x, y):
    """Regression coefficients.
    
    Returns the least squares regression coefficients.
    
    Parameters
    ----------
    x : array_like
        Input array of predictor(s)
    y : array_like
        Input array of response(s)
    
    Returns
    -------
    r : ndarray
        Least squares estimates of the regression coefficients
    """
    
    inv = np.linalg.inv(np.matmul(x.T, x))
    r = np.matmul(inv, np.matmul(x.T, y))
    return r

def reg_coeff_cov(x, y):
    """Covariance matrix of regression coefficients.
    
    Returns the covariance matrix for the regression coefficients estimated using ordinary least
    squares.
    
    Parameters
    ---------- 
    x : array_like
        Input array of predictor(s)
    y : array_like
        Input array of response(s)
    
    Returns
    -------
    r : ndarray
        Covariance matrix of regression coefficients
    """
    
    s2 = mse(x, y) # MLE of error variance
    inv = np.linalg.inv(np.matmul(x.T, x))
    r = s2 * inv
    return r
    
def lse(x, y):
    """Least squares estimates of response.
    
    Estimates the responses using the predictors. The estimates are such that
    they minimize the square error between the estimates and the given responses.
    
    Parameters
    ----------
    x : array_like
        Input array of predictor(s)
    y : array_like
        Input array of response(s)
    """
    return np.matmul(proj_mat(x), y)

def weighted_lse(x, y, w):
    """Return the weighted least squares estimates.
    """
    
    mat1 = np.linalg.inv(np.matmul(x.T, np.matmul(w, x)))
    mat2 = np.matmul(x.T, np.matmul(w, y))
    return np.matmul(mat1, mat2)

def weighted_pred(x, y, w):
    """Return predicted response via weighted least squares estimates.
    """
    return np.matmul(x, weighted_lse(x, y, w))

def weighted_res(x, y, w):
    """Return the residuals for the weighted least squares estimates of the response.
    """
    return y - weighted_pred(x, y, w)

def res(x, y):
    """
    Residuals for least squares estimates.
    
    Parameters
    ----------
    x : array_like
        input array of predictors
    y : array_like
        input array of responses
    """
    return y - lse(x, y)

def del_res(x, y):
    """Deleted residuals for least squares estimates.
    
    Parameters
    ----------
    x : array_like
        input array of predictors
    y : array_like
        input array of responses
    """
    return (res(x, y).T / (1 - np.diag(proj_mat(x)))).T

def sse(x, y): 
    """Residual sum of squares.
    
    Parameters
    ----------
    x : array_like
        input array of predictors
    y : array_like
        input array of responses
    """
    return np.matmul(res(x, y).T, res(x, y))[0,0]
    
def mse(x, y):
    """Mean squared error.
    
    Parameters
    ----------
    x : array_like
        input array of predictors
    y : array_like
        input array of responses
    """
    return sse(x, y) / (len(x) - len(x[0]))

def ssto(y):
    """Total sum of squares of an array.
    
    Parameters
    ----------
    y : array_like
        input array of responses
    """
    return sse(np.ones((len(y), 1)), y)

def coeff_det(x, y):
    """Coefficient of multiple determination.
    
    Parameters
    ----------
    x : array_like
        input array of predictors
    y : array_like
        input array of responses
    """
    return 1 - (sse(x, y) / ssto(y))
    
    
def adj_coeff_det(x, y):
    """Adjust coefficient of multiple determination.
    
    Parameters
    ----------
    x : array_like
        input array of predictors
    y : array_like
        input array of responses
    """
    return 1 - ((mse(x, y) / ssto(y)) * (len(y) - 1))
    
def Cp(x_full, x_red, y):
    """Mallows' C_p criterion.
    
    Parameters
    ----------
    x_full : array_like
        input array of predictors corresponding to the full model
    x_red : array_like
        input array of predictors corresponding to the reduced model (subset)
    y : array_like
        input array of responses
    """
    n = len(y)
    p = len(x_red[0])
    return sse(x_red, y) / mse(x_full, y) - (n - 2*p)
    
def press(x, y):
    """Prediction sum of squares.
    
    Parameters
    ----------
    x : array_like
        input array of predictors
    y : array_like
        input array of responses
    """
    return np.matmul(del_res(x, y).transpose(), del_res(x, y))[0,0]
    
def sbc(x, y):
    """Schwarz' Bayesian criterion.
    
    Parameters
    ----------
    x : array_like
        input array of predictors
    y : array_like
        input array of responses
    """
    
    n = len(x)
    p = len(x[0])+1
    return len(x) * np.log(sse(x, y)) - n * np.log(n) + np.log(n) * p
    
def aic(x, y):
    """Akaike's information criterion.
    
    Parameters
    ----------
    x : array_like
        input array of predictors
    y : array_like
        input array of responses
    """
    
    n = len(x)
    p = len(x[0])+1
    return len(x) * np.log(sse(x, y)) - n * np.log(n) + 2 * p

