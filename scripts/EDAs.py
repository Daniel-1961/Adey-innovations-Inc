import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def univariate_analysis(df, column):
    """Plots histogram and KDE for a given numerical column."""
    plt.figure(figsize=(8, 4))
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(f'Univariate Analysis of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def bivariate_analysis(df, col1, col2, plot_type='box'):
    """Plots bivariate analysis between two columns using box plot or scatter plot."""
    plt.figure(figsize=(8, 4))
    if plot_type == 'box':
        sns.boxplot(x=df[col1], y=df[col2])
    elif plot_type == 'scatter':
        sns.scatterplot(x=df[col1], y=df[col2])
    plt.title(f'Bivariate Analysis: {col1} vs {col2}')
    plt.show()
