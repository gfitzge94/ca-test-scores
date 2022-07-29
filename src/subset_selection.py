"""A library of procedures for subset selection.

This module requires that regression.py, numpy, and pandas be installed with the 
Python environment that you are exporting this module to.

Functions
---------
all_crit_values(x, y, max_p)
    All criteria for model selection for every subset upto a given size.
best_crit_values(x, y, max_p)
    Best values for model selection criteria for every subset upto a given size.
best_subsets(x, y, max_p)
    Best subsets for each model selection criterion.
for_step_reg(x, y, al_to_enter, al_to_exit)
    Subset obtained from the forward stepwise regression procedure.
sel_crit_plot(x, y, max_p)
    Panel of figures corresponding to the different subset selection criteria.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import t
import src.regression as reg

def all_crit_values(x, y, max_subset_size):
    """Returns a DataFrame containing all model selection criterion values
    for every subset upto a specified size.
    
    Parameters
    ----------
    x : array-like
        Predictors
    y : array-like
        Response
    max_p : int
        Maximum number of predictors to be considered in a subset
        
    Returns
    -------
    df : DataFrame
        DataFrame consisting of the values of all selection criteria for 
        every possible subset upto a given size.
    """
    
    # DataFrame with columns for selection criteria
    df = pd.DataFrame(columns = ['subset',
                                 'p',
                                 'Rp^2',
                                 'Rap^2',
                                 'Cp',
                                 'AICp',
                                 'SBCp',
                                 'PRESSp'])
    
    # Index all possible subsets upto max_subset_size
    subset_indices = []
    n_pred = len(x[0])
    for subset_size in range(1, max_subset_size + 1):
        for subset_index in combinations(range(n_pred), subset_size):
            if 0 in subset_index:
                subset_indices.append(subset_index)
        
    # Selection criteria for all subsets
    df_index = 0
    for i in subset_indices:
        s = x[:, i] # predictor subset
    
        sv = ''
        for j in i:
            sv += 'X_{}, '.format(j) # variable names
        
        df.loc[df_index] = {'subset': sv,
                            'p': len(i),
                            'Rp^2': reg.coeff_det(s, y),
                            'Rap^2': reg.adj_coeff_det(s, y),
                            'Cp': reg.Cp(x, s, y),
                            'AICp': reg.aic(s, y),
                            'SBCp': reg.sbc(s, y),
                            'PRESSp': reg.press(s, y)}
        df_index += 1
    
    return df.round(decimals={
        'p': 0,
        'Rp^2': 3,
        'Rap^2': 3,
        'Cp': 3,
        'AICp': 3,
        'SBCp': 3,
        'PRESSp': 3})

def best_crit_values(x, y, max_p):
    """Returns DataFrame containing the best values for each model selection criterion 
    for every subset upto a given size.
    
    Parameters
    ----------
    x : array_like
        input array of predictors
    y : array_like
        input array of responses
    max_p : int
        maximum number of predictors to be considered in a subset
        
    Returns
    -------
    out : DataFrame
        DataFrame consisting of the best criteria selection values for each
        size subset upto a given limit.
    """
    
    df = all_crit_values(x, y, max_p)
    df.drop('subset', inplace=True, axis=1) # Drop subset column
    #print(df.head())
    
    g = df.groupby('p')
    #print(g)
    
    out = g.agg({'Rp^2': 'max',
                 'Rap^2': 'max',
                 'Cp': 'min',
                 'AICp': 'min',
                 'SBCp': 'min',
                 'PRESSp': 'min'})
    
    return out

def best_subsets(x, y, max_p):
    """Identifies the best subsets for each model selection criterion and each size subset.
    
    Parameters
    ----------
    x : array_like
        input array of predictors
    y : array_like
        input array of responses
    max_p : int
        maximum number of predictors to be considered in a subset
        
    Returns
    -------
    df : DataFrame
        DataFrame consisting of the best subsets for each model selection
        criterion and each subset size upto a given limit.
    """

    df = pd.DataFrame(columns = ['p',
                                 'Rp^2',
                                 'Rap^2',
                                 'Cp',
                                 'AICp',
                                 'SBCp',
                                 'PRESSp'])
    
    crit_list = list(df.columns.drop('p'))
    #print(crit_list)
    
    all_criterion_values = all_crit_values(x, y, max_p)
    #print(all_criterion_values.head())
    
    best_criterion_values = best_crit_values(x, y, max_p)
    #print(best_criterion_values.loc[1, '$SSE_p$'])
    
    for p in range(1, max_p + 1):
        df.loc[p, 'p'] = p
        for crit in crit_list:
            best_crit_val = best_criterion_values.loc[p, crit]
            #print(best_crit_val)
            
            mask = all_criterion_values[crit] == best_crit_val
            index = all_criterion_values.index[mask].tolist()[0]
            #print(index)
            
            #print(all_criterion_values.loc[index, 'subset'])
            
            df.loc[p, crit] = all_criterion_values.loc[index, 'subset']
            
    return df

def forward_step(x, y, al_to_enter=0.1, al_to_exit=0.15):
    """Obtain "best" subset using the forward stepwise regression procedure.
    """
    # Check for valid inputs
    if al_to_enter <= 0: 
        return print("al_to_enter must be greater than 0.")
    elif al_to_enter >= al_to_exit: 
        return print("al_to_enter must be less than al_to_exit.")
    elif al_to_exit >= 1: 
        return print("al_to_exit must be less than 1.")
    
    n_pred = len(x[0])
    x_in = [0] # Variables in the model. Start with the intercept term.
    x_out = [i for i in range(1, n_pred)] # Variables out of the model
    
    p_values = {}
    
    for k in x_out: # each predictor
        subset = np.append(x[:, x_in], x[:, [k]], axis=1)
        sse_R = reg.sse(x[:, x_in], y)
        sse_F = reg.sse(subset, y)
        t_stat = np.sqrt((sse_R - sse_F) / reg.mse(subset, y))
        df0 = len(x) - len(subset[0]) - 1
        p_val = 2 * (1 - t.cdf(np.abs(t_stat), df=df0))
        p_values[p_val] = k
    
    if min(p_values) < al_to_enter:
        x_in.append(p_values[min(p_values)]) # add kth column to x_in
        x_out.remove(p_values[min(p_values)]) # delete kth column from x_out
        print("Add X_{}.".format(p_values[min(p_values)]))
              
    else:
        x_in.sort()
        return print("No variables could be added to the model.")
              
    while x_out != []:
        
        # add step
        p_values = {}
    
        for k in x_out: # each predictor
            subset = np.append(x[:, x_in], x[:, [k]], axis=1)
            sse_R = reg.sse(x[:, x_in], y)
            sse_F = reg.sse(subset, y)
            t_stat = np.sqrt((sse_R - sse_F) / reg.mse(subset, y))
            df0 = len(x) - len(subset[0]) - 1
            p_val = 2 * (1 - t.cdf(np.abs(t_stat), df=df0))
            p_values[p_val] = k
  
        if min(p_values) < al_to_enter:
            x_in.append(p_values[min(p_values)]) # add kth column to x_in
            x_out.remove(p_values[min(p_values)]) # delete kth column from x_out
            print("Add X_{}.".format(p_values[min(p_values)]))
              
        else:
            x_in.sort()
            return print("Best subset: ", ["X_{}".format(i) for i in x_in])
        
        # remove step
        p_values = {}
        for k in x_in: # each predictor
            subset = x_in.copy()
            subset.remove(k)
            sse_R = reg.sse(x[:, subset], y)
            sse_F = reg.sse(x[:, x_in], y)
            t_stat = np.sqrt((sse_R - sse_F) / reg.mse(x[:, subset], y))
            df0 = len(x) - len(x_in) - 1
            p_val = 2 * (1 - t.cdf(np.abs(t_stat), df=df0))
            p_values[p_val] = k
  
        if max(p_values) > al_to_exit:
            x_in.remove(p_values[max(p_values)])
            x_out.append(p_values[max(p_values)])
            print("Remove X_{}.".format(p_values[max(p_values)]))       
    
    x_in.sort()
    return print("Best subset: ", ["X_{}".format(i) for i in x_in])
    
def sel_crit_plot(x, y, max_p):
    """Plots selection criteria for all subsets.
    
    Produces a panel of six plots, one for each subset selection criterion.
    
    Parameters
    ----------
    x: array_like
        Array of predictors
        
    y: array_like
        Array of responses
       
    max_p: int
        Maximum number of predictors in a subset
    
    Return
    ------
    r:
        Figure with six plots
    """
    
    all_values = all_crit_values(x, y, max_p)
    best_values = best_crit_values(x, y, max_p)
    
    fig, ax = plt.subplots(3, 2, sharex='col', tight_layout=True)
    fig.set_size_inches(12, 12)
    #fig.subplots_adjust(wspace=0.5)
    fig.suptitle("Selection criteria plots")
    
    columns = [['Rp^2', 'Rap^2'],
               ['Cp', 'AICp'],
               ['SBCp', 'PRESSp']]
    
    for i in range(0,3):
        for j in range(0,2):
            c = columns[i][j] # current column
            ax[i,j].plot(all_values['p'], all_values[c], '.')
            ax[i,j].plot(best_values.index, best_values[c])
            ax[i,j].set_ylabel(c)
            ax[i,j].set_xlabel("Number of predictors")
            
    return fig

            