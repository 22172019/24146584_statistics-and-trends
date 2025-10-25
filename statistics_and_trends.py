

# Automobile Data Analysis Report
"""

"""
This is the template file for the statistics and trends assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew, kurtosis
from google.colab import drive
drive.mount('/content/drive')


def plot_relational_plot(df):
    """
    Generate relational plots (scatter: horsepower vs price)
    and save as 'relational_plot.png'.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=df, x='horsepower', y='price', hue='fuel-type',
                    style='fuel-type', s=120, palette='Set2',
                    edgecolor='black', alpha=0.85, ax=ax)
    plt.title('Horsepower vs Price by Fuel Type')
    plt.xlabel('Horsepower')
    plt.ylabel('Price')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('relational_plot.png')
    plt.close()
    return


def plot_categorical_plot(df):
    """
    Generate categorical plots (bar: fuel-type vs avg price)
    and save as 'categorical_plot.png'.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    avg_fuel = df.groupby('fuel-type')['price'].mean().sort_values()
    sns.barplot(x=avg_fuel.index, y=avg_fuel.values, palette='Set2',
                edgecolor='black', ax=ax)
    plt.title('Fuel Type vs Average Price')
    plt.ylabel('Average Price')
    plt.xlabel('Fuel Type')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('categorical_plot.png')
    plt.close()
    return


def plot_statistical_plot(df):
    """
    Generate statistical plots (boxplot: price by fuel-type)
    and save as 'statistical_plot.png'.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.boxplot(x='fuel-type', y='price', data=df, palette='Set2',
                linewidth=1.5, showmeans=True,
                meanprops={"marker": "o",
                           "markerfacecolor": "white",
                           "markeredgecolor": "black",
                           "markersize": 10},
                ax=ax)
    sns.stripplot(x='fuel-type', y='price', data=df, color='black',
                  size=4, jitter=True, alpha=0.6, ax=ax)
    plt.title('Boxplot: Price by Fuel Type')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('statistical_plot.png')
    plt.close()
    return


def statistical_analysis(df, col: str):
    """
    Compute mean, standard deviation, skewness, and excess kurtosis
    for the given numeric column.
    """
    mean = df[col].mean()
    stddev = df[col].std()
    skew_val = skew(df[col])
    excess_kurtosis = kurtosis(df[col])
    return mean, stddev, skew_val, excess_kurtosis


def preprocessing(df):
    """
    Clean and preprocess the dataset:
    - Handle missing values
    - Convert numeric columns
    - Remove outliers
    - Display preview and statistics
    """
    df = df.copy()
    df.replace('?', pd.NA, inplace=True)
    df.dropna(subset=['price', 'horsepower', 'fuel-type'], inplace=True)

    numeric_cols = ['price', 'horsepower', 'city-mpg',
                    'highway-mpg', 'engine-size', 'curb-weight']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numeric_cols, inplace=True)

    # Handle outliers for price and horsepower
    for col in ['price', 'horsepower']:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]

    print("Data Preview:\n", df.head())
    print("\nSummary Statistics:\n", df.describe())
    print("\nCorrelation Matrix:\n", df.corr(numeric_only=True))
    return df


def writing(moments, col):
    """
    Display computed statistical moments with interpretation.
    """
    print(f'\nFor the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, '
          f'Excess Kurtosis = {moments[3]:.2f}.')

    # Interpret skewness and kurtosis
    if moments[2] > 0:
        skew_type = "right-skewed"
    elif moments[2] < 0:
        skew_type = "left-skewed"
    else:
        skew_type = "symmetrical"

    if moments[3] < 0:
        kurt_type = "platykurtic (flatter tails)"
    elif -2 <= moments[3] <= 2:
        kurt_type = "mesokurtic (normal-like)"
    else:
        kurt_type = "leptokurtic (heavy tails)"

    print(f"The data is {skew_type} and {kurt_type}.")
    return


def main():
    """
    Main execution function for the statistics and trends assignment.
    """
    df = pd.read_csv('data.csv')  # replace with your data path
    df = preprocessing(df)
    col = 'price'

    # Generate plots
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    # Perform statistical analysis
    moments = statistical_analysis(df, col)
    writing(moments, col)
    return


if __name__ == '__main__':
    main()

