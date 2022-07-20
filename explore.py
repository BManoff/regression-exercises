
from pydataset import data # importing librabries
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import env

def plot_variable_pairs(df):
   sns.pairplot(df, kind = "reg", plot_kws={'line_kws':{'color':'red'}})

def plot_categorical_and_continuous_vars(df, categorical_vars, continuous_vars):
    for count in continuous_vars:
        for cat in categorical_vars:
            _, ax = plt.subplots(1,3,figsize=(20,8))
            p = sns.stripplot(data = df, x=cat, y=count, ax=ax[0], s=1)
            p.axhline(df[count].mean())
            p = sns.boxplot(data = df, x=cat, y = count, ax=ax[1])
            p.axhline(df[count].mean())
            p = sns.violinplot(data = df, x=cat, y=count, hue = cat, ax=ax[2])
            p.axhline(df[count].mean())
            plt.suptitle(f'{count} by {cat}', fontsize = 18)
            plt.show()