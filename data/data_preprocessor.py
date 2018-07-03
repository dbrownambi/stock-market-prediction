import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression


def get_cleaned_data(datafile):
    df = pd.read_csv(datafile)
    df['Next_Day'] = pd.Series(df['Close'], index=df.index)
    df['CloseA'] = pd.Series(df['Close'], index=df.index)
    df['CloseB'] = pd.Series(df['Close'], index=df.index)
    df['CloseC'] = pd.Series(df['Close'], index=df.index)
    df['CloseD'] = pd.Series(df['Close'], index=df.index)
    print(df.head())
    for i in range(4, len(df) - 1):
        df.at[i, 'Next_Day'] = df.iloc[i + 1]['Close']
        df.at[i, 'CloseA'] = df.iloc[i - 1]['Close']
        df.at[i, 'CloseB'] = df.iloc[i - 2]['Close']
        df.at[i, 'CloseC'] = df.iloc[i - 3]['Close']
        df.at[i, 'CloseD'] = df.iloc[i - 4]['Close']

    df = df[["CloseA", "CloseB", "CloseC", "CloseD", "Close", "Next_Day"]].copy()
    df = df.iloc[5:]
    return df


