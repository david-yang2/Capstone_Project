import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def merge_dfs(df1, df2, column = 'date'):
    '''
    INPUT:
    df1 = trips df
    df2 = weather df

    OUTPUT: combined df

    '''
    #weather dates are type str
    #need to convert to datetime before we can merge with trips df
    #because dates column in trips df is datetime
    df2[column] = pd.to_datetime(df2[column])
    df2[column] = df2[column].dt.date
    combined_df = pd.merge(df1,df2, on='date', how='left')
    return combined_df

