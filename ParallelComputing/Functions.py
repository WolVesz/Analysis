from minepy import MINE
import pandas as pd

def Maximal_Compute(df, col, tic = False):
    #http://web.mit.edu/dnreshef/www/websiteFiles/Papers/AOAS-2018.pdf
    mine = MINE(alpha=0.6, c=15)
    
    col_dict = dict()
    col_dict['index'] = col
    
    if tic:
        tic_dict = dict()
        tic_dict['index'] = col
    
    for column in df.columns:
        mine.compute_score(df[col], df[column])
        col_dict[column] = mine.mic()

        if tic:
            tic_dict[column] = mine.tic()
    if tic:
        return (col_dict, tic_dict)
    else:
        return col_dict
