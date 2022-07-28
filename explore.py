
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

def graph_features(df):
    # figure size
    plt.rcParams["figure.figsize"] = (16, 5)
    # pick the columns to plot
    cols = ["bedroomcnt", "bathroomcnt", "calculatedfinishedsquarefeet", "yearbuilt"]
    # plt.subplot(row X col, where?)
    fig, axes = plt.subplots(1, 4, sharey=True)
    # run throught the columns and plot the distribution
    for i, col in enumerate(cols):
        # Title with column name.
        axes[i].set_title(col)
        # Display lmplot for column.
        sns.regplot(
            data=df,
            x=col,
            y="taxvaluedollarcnt",
            line_kws={"color": "green"},
            ax=axes[i],
        )