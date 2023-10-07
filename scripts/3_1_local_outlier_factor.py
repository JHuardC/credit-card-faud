"""
Create local outlier factor feature
"""
### Imports
import pathlib as pl
import dotenv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import joblib

### Constants
PROJECT_PATH = pl.Path(dotenv.find_dotenv()).absolute().parent

X_train = pd.read_parquet(
    PROJECT_PATH.joinpath('data', 'train-test-pack', 'X_train.pqt')
)

standardizer = StandardScaler()
X_train = pd.DataFrame(
    standardizer.fit_transform(X_train),
    columns = X_train.columns
)

outlier_factor = LocalOutlierFactor(
    n_neighbors = 20,
    algorithm = 'ball_tree',
    n_jobs = -1
)
outlier_factor.fit(X_train)

joblib.dump(
    outlier_factor,
    PROJECT_PATH.joinpath('outputs', 'eda', 'local-outlier-factor.joblib')
)