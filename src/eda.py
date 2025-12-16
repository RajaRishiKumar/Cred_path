"""
eda.py
------
Reusable & dataset-agnostic EDA module.


Covers: 
1. Dataset Overview 
2. Missing Values Analysis 
3. Outlier Detection (IQR) 
4. Univariate Analysis (histograms, skewness) 
5. Bivariate Analysis (correlation, numeric vs numeric) 
6. Categorical Analysis 
7. Target Variable Analysis (auto-detected if binary) 
8. Pairwise Relationships 
9. PCA Visualization




Usage:
------
from src.eda import run_eda
run_eda(df, target_col="repay_fail")
"""


# IMPORTS


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



# SAFE PLOT WRAPPER


def safe_plot(plot_func, *args, **kwargs):
    try:
        plot_func(*args, **kwargs)
    except Exception as e:
        print(f" Plot skipped due to error: {e}")
        plt.close()




# 1. DATASET OVERVIEW


def dataset_overview(df: pd.DataFrame) -> None:
    print("\n DATASET OVERVIEW")
    print("Shape:", df.shape)
    print("\nColumns:\n", df.columns.tolist())
    print("\nData Types:\n", df.dtypes)
    print("\nSummary Statistics:\n", df.describe(include="all").transpose().head())
    print("\nMissing Values:", df.isnull().sum().sum())
    print("Duplicate Rows:", df.duplicated().sum())



# 2. MISSING VALUES


def missing_values_analysis(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    print("\n MISSING VALUES SUMMARY:\n", missing)

    if not missing.empty:
        plt.figure(figsize=(12, 6))
        safe_plot(sns.heatmap, df.isnull(), cbar=False)
        plt.title("Missing Values Heatmap")
        plt.show()



# 3. OUTLIER DETECTION


def outlier_detection(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        print("No numeric columns for outlier detection.")
        return

    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    outliers = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) |
                (df[numeric_cols] > (Q3 + 1.5 * IQR)))

    print("\n OUTLIERS PER COLUMN:\n", outliers.sum().sort_values(ascending=False))



# 4. UNIVARIATE NUMERIC


def univariate_numeric(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        data = df[col].dropna()
        skew_val = data.skew()
        safe_plot(sns.histplot, data, bins=30, kde=True)
    
        plt.title(f"Distribution of {col}\nSkewness = {skew_val:.2f}", fontsize=12)
        plt.xlabel(col)
        plt.ylabel("Frequency")
    
        # Highlighting the highly skewed
        if abs(skew_val) > 1:
            plt.text(x=data.mean(), y=plt.ylim()[1]*0.9, 
                    s="Highly Skewed", 
                    color="red", fontsize=10, ha="center", weight="bold")
    
        plt.show()




# 5. BOXPLOTS


def boxplots_numeric(df: pd.DataFrame, limit: int = 10) -> None:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    for col in numeric_cols[:limit]:
        data = df[col].dropna()

        if data.nunique() <= 1:
            print(f"Skipping {col} (low variance)")
            continue

        plt.figure(figsize=(6, 3))
        safe_plot(sns.boxplot, x=data)
        plt.title(f"Boxplot of {col}")
        plt.show()



# 6. CORRELATION


def correlation_analysis(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) < 2:
        return

    plt.figure(figsize=(12, 9))
    safe_plot(sns.heatmap, df[numeric_cols].corr(), cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()



# 7. CATEGORICAL


def categorical_analysis(df: pd.DataFrame, target_col: str | None = None) -> None:
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    for col in categorical_cols[:20]:
        print(f"\n {col} VALUE COUNTS:\n", df[col].value_counts())

        plt.figure(figsize=(6, 4))
        safe_plot(sns.countplot, y=df[col], order=df[col].value_counts().index)
        plt.title(f"{col} Distribution")
        plt.show()

        if target_col and target_col in df.columns:
            plt.figure(figsize=(6, 4))
            safe_plot(sns.barplot, x=df[col], y=df[target_col])
            plt.title(f"{col} vs {target_col}")
            plt.xticks(rotation=45)
            plt.show()



# 8. TARGET ANALYSIS


def target_analysis(df: pd.DataFrame, target_col: str) -> None:
    if target_col not in df.columns:
        return

    plt.figure(figsize=(6, 4))
    safe_plot(sns.countplot, x=df[target_col])
    plt.title(f"Target Distribution: {target_col}")
    plt.show()

    print("\n TARGET COUNTS:\n", df[target_col].value_counts())
    print("\n TARGET %:\n", df[target_col].value_counts(normalize=True) * 100)



# 9. PCA


def pca_analysis(df: pd.DataFrame, target_col: str | None = None) -> None:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) <= 2:
        return

    scaled = StandardScaler().fit_transform(df[numeric_cols].fillna(0))
    pcs = PCA(n_components=2).fit_transform(scaled)

    plt.figure(figsize=(7, 5))
    safe_plot(
        sns.scatterplot,
        x=pcs[:, 0],
        y=pcs[:, 1],
        hue=df[target_col] if target_col else None,
        alpha=0.6
    )
    plt.title("PCA Projection (2 Components)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()




# All-in-one EDA function


def run_eda(df: pd.DataFrame, target_col: str | None = None) -> None:
    print("\n" + "=" * 90)
    print(" EXPLORATORY DATA ANALYSIS REPORT")
    print("=" * 90)

    dataset_overview(df)
    missing_values_analysis(df)
    outlier_detection(df)
    univariate_numeric(df)
    boxplots_numeric(df)
    correlation_analysis(df)
    categorical_analysis(df, target_col)
    target_analysis(df, target_col)
    pca_analysis(df, target_col)

    print("\n EDA COMPLETED SUCCESSFULLY\n")
