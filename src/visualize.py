import matplotlib.pyplot as plt
import seaborn as sns
from math import floor
from math import ceil

def boxplots(data, vars=None):
    """Create figure with horizontal box plots for each column for the given variables.
    
    If no variables are specified, create boxplots for all columns of the data.
    """
    
    if vars == None:
        vars = data.columns.tolist()
    
    n_vars = len(vars)
    
    fig, ax = plt.subplots(n_vars)
    fig.set_size_inches(8, n_vars)
    
    i = 0 # Initialize subplot counter
    for var in vars:
        sns.boxplot(x=data[var], ax=ax[i]);
        i += 1
        
    fig.tight_layout()
    
    return ax

def scatterplots(x, y):
    """Create figure with scatterplots for each input and the output.
    """
    
    n_vars = len(x.columns)
    n_rows = ceil(n_vars / 3)
    
    fig, ax = plt.subplots(n_rows, 3)
    fig.set_size_inches(12, n_vars)
    
    i = 0
    for var in x.columns:
        row = i % 3
        col = floor(i / 3)
        sns.regplot(x=x[var], y=y, lowess=True, line_kws={'color': 'r'}, ax=ax[col, row])
        i += 1
        
    fig.tight_layout()
    
    return ax



