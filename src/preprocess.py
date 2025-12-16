

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold



# 1. MISSING VALUE HANDLING

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    if len(cat_cols) > 0:
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df



# 2. TYPE CONVERSION UTILITIES

def convert_percent_to_float(df: pd.DataFrame, percent_cols: list) -> pd.DataFrame:
    df = df.copy()

    for col in percent_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .replace("nan", np.nan)
                .astype(float)
            )

    return df




def parse_datetime_columns(df: pd.DataFrame, datetime_cols: list) -> pd.DataFrame:
    df = df.copy()

    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            # Extract year, month,date as numeric features
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            print(f"Parsed datetime column '{col}' into '{col}_year', '{col}_month'and '{col}_day'.Also, if u want u can drop the original column as well")
    return df

import pandas as pd
import numpy as np
from typing import List, Optional

def convert_floats_to_float32(df: pd.DataFrame) -> pd.DataFrame:

    float64_cols = df.select_dtypes(include=['float64']).columns
    print(f"Downcasting {len(float64_cols)} float64 columns to float32.")
    
    for col in float64_cols:
        df[col] = df[col].astype('float32')

    return df


def downcast_integers(df: pd.DataFrame) -> pd.DataFrame:

    int_cols = df.select_dtypes(include=['int64']).columns
    print(f"Downcasting {len(int_cols)} int64 columns to smaller integer types.")

    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')
        
    return df





# 3. ENCODING

def one_hot_encode(df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not cat_cols:
        return df

    df = pd.get_dummies(df, columns=cat_cols, drop_first=drop_first)
    return df


def label_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()
    le = LabelEncoder()

    le_dict = {}
    print("store encoders if needed later, in le_dict with column names as keys")


    for col in columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le 
    return df


# 3b. BOOLEAN TO INTEGER CONVERSION
def convert_bool_to_int(df: pd.DataFrame):
    df = df.copy()
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df



"""
# 4. OUTLIER HANDLING

def handle_outliers_iqr(df: pd.DataFrame, method: str = "cap") -> pd.DataFrame:

    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        if method == "cap":
            df[col] = np.clip(df[col], lower, upper)

        elif method == "remove":
            df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df


def handle_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        z = (df[col] - df[col].mean()) / df[col].std()
        df = df[z.abs() <= threshold]

    return df



# 5. FEATURE SCALING

def scale_features(df: pd.DataFrame, method: str = "standard"):
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns

    if not num_cols.any():
        return df, None

    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, scaler



# 6. LOW VARIANCE REMOVAL

def remove_low_variance(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns

    if len(num_cols) == 0:
        return df

    selector = VarianceThreshold(threshold=threshold)
    selected = selector.fit_transform(df[num_cols])

    selected_cols = num_cols[selector.get_support()]
    df = df[selected_cols.tolist() + df.select_dtypes(exclude=[np.number]).columns.tolist()]

    return df



# 7. SKEWNESS CORRECTION

def log_transform_skewed(df: pd.DataFrame, skew_threshold: float = 1.0) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        if df[col].skew() > skew_threshold and (df[col] >= 0).all():
            df[f"{col}_log"] = np.log1p(df[col])

    return df

"""

# 8. All-in-One Preprocessing Function

def preprocess(
    df: pd.DataFrame,
    percent_cols=None,
    datetime_cols=None,
    encode=True,
    #outlier_method=None,
    #scale_method=None
):
    df = handle_missing_values(df)

    if percent_cols:
        df = convert_percent_to_float(df, percent_cols)

    if datetime_cols:
        df = parse_datetime_columns(df, datetime_cols)

    if encode:
        df = one_hot_encode(df)

    return df


"""    if outlier_method == "iqr_cap":
        df = handle_outliers_iqr(df, method="cap")

    elif outlier_method == "iqr_remove":
        df = handle_outliers_iqr(df, method="remove")

    elif outlier_method == "zscore":
        df = handle_outliers_zscore(df)

    scaler = None
    if scale_method:
        df, scaler = scale_features(df, method=scale_method)"""

 #, scaler
