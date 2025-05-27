import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split # type: ignore
from typing import Tuple, Sequence
from io import StringIO
from scipy.io import arff
from pathlib import Path

def drop_highly_correlated(X: pd.DataFrame, threshold: float = 0.9) -> list[str]:
    # Compute absolute correlations and identify upper triangle of the matrix 
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [c for c in upper.columns if (upper[c] > threshold).any()]

def clean_and_split(
    df: pd.DataFrame,
    class_column_name: str,
    n: int = 5,
    select_option: int = 0,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select up to n samples per class (or recode classes), split into
    train/test (stratified), then drop any features in train whose
    pairwise correlation > 0.9, removing the same from test.
    n ≤ 0 means select all rows for each class.
    """
    # declare that we may select ints or strs
    class_values_to_select: Sequence[int | str]
    # choose which class values to include
    if select_option == 0:
        # ensure a plain Python list of ints/strs
        class_values_to_select = df[class_column_name].unique().tolist()
    elif select_option == 1:
        class_values_to_select = ['CHAT', 'VPN-CHAT', 'MAIL', 'VPN-MAIL']
    elif select_option == 2:
        df[class_column_name] = (
            df[class_column_name].str.contains('VPN').astype(int)
        )
        class_values_to_select = [1, 0]
    else:
        raise ValueError(
            "select_class must be 0 (all), 1 (subset), or 2 (VPN vs non-VPN)"
        )

    # sample up to n rows per class
    selected = []
    for v in class_values_to_select:
        subset = df[df[class_column_name] == v]
        if n <= 0:
            selected.append(subset)
        else:
            selected.append(subset.head(n))
    df_sel = pd.concat(selected, ignore_index=True)

    # stratified split
    train, test = train_test_split(
        df_sel,
        test_size=test_size,
        random_state=random_state,
        stratify=df_sel[class_column_name]
    )

    # --- NEW: ensure no sample is in both train and test ---
    overlap = set(train.index) & set(test.index)
    if overlap:
        raise ValueError(f"Train/test split overlap detected at indices: {overlap}")
    # --------------------------------------------------------

    # Separate features and target for train and test
    y_train = train[class_column_name]
    X_train = train.drop(columns=[class_column_name])
    to_drop = drop_highly_correlated(X_train)  # using new helper function

    # drop from train and reattach target
    X_train_clean = X_train.drop(columns=to_drop)
    train_clean = pd.concat([X_train_clean, y_train], axis=1)

    # drop from test and reattach target (modified)
    test_clean = test.drop(columns=to_drop)

    return train_clean, test_clean

def dataframe_summary(df:pd.DataFrame)->str:
    buffer = StringIO()
    df.info(buf=buffer)
    return (
        f"Shape: {df.shape}\n"
        f"Memory: {df.memory_usage(deep=True).sum()/1e6:.2f} MB\n"
        f"Missing Values:\n{df.isna().sum()}\n"
        f"Info:\n{buffer.getvalue()}"
    )

def select_random_columns(df: pd.DataFrame, mandatory_col: str, n_random: int, random_state: int = 42)->pd.DataFrame:
    if mandatory_col not in df.columns:
        raise ValueError(f"Mandatory column '{mandatory_col}' not in DataFrame.")

    # Use Index operations directly
    other_cols = df.columns.drop(mandatory_col)
    n_random = min(n_random, len(other_cols))
    rng = np.random.default_rng(random_state)
    random_cols = rng.choice(other_cols, size=n_random, replace=False)

    selected = [mandatory_col] + list(random_cols)
    return df[selected]

def load_dataframe(file_path: Path) -> pd.DataFrame:
    """
    Load a DataFrame from a CSV or ARFF file.

    Args:
        file_path:Path: Path to the input file (.csv or .arff)

    Returns:
        pd.DataFrame: Loaded DataFrame

    Raises:
        ValueError: If file extension is unsupported
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    elif path.suffix.lower() == ".arff":
        data, _ = arff.loadarff(path)
        df = pd.DataFrame(data)
        # Decode byte strings if needed
        for col in df.select_dtypes(include=[object]):
            df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
        return df

    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
