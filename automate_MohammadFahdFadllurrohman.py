import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)


def load_data(csv_path: str = 'iris.csv') -> pd.DataFrame:
    """Load dataset dari CSV atau langsung dari sklearn jika file tidak ada."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        log.info(f"Dataset dimuat dari file: {csv_path} | Shape: {df.shape}")
    else:
        log.info("File CSV tidak ditemukan, memuat dari sklearn...")
        iris = load_iris()
        df = pd.DataFrame(iris.data,
                          columns=['sepal_length', 'sepal_width',
                                   'petal_length', 'petal_width'])
        df['species'] = iris.target
        df.to_csv(csv_path, index=False)
        log.info(f"Dataset disimpan ke {csv_path} | Shape: {df.shape}")
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validasi dan bersihkan data."""
    log.info(f"Missing values: {df.isnull().sum().sum()}")
    log.info(f"Duplikat: {df.duplicated().sum()}")

    df = df.drop_duplicates().reset_index(drop=True)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            log.info(f"Imputasi median pada kolom: {col}")

    log.info(f"Shape setelah cleaning: {df.shape}")
    return df


def preprocess(df: pd.DataFrame, target_col: str = 'species'):
    """Preprocessing: scaling dan train-test split."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    log.info(f"Normalisasi selesai. Mean ~0, Std ~1")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def save_data(X_train, X_test, y_train, y_test,
              output_dir: str = 'nadataset_preprocessing') -> None:
    """Simpan hasil preprocessing ke folder output."""
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)

    log.info(f"Data tersimpan di '{output_dir}/'")
    log.info(f"Files: {os.listdir(output_dir)}")


def main():
    log.info("=" * 50)
    log.info("Memulai automated preprocessing...")
    log.info("=" * 50)

    df = load_data()
    df = validate_data(df)
    X_train, X_test, y_train, y_test = preprocess(df)
    save_data(X_train, X_test, y_train, y_test)

    log.info("Preprocessing selesai!")


if __name__ == '__main__':
    main()
