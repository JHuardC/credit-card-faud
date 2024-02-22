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

df_eda = pd.read_parquet(
    PROJECT_PATH.joinpath('data', 'eda-pack', 'df_eda.pqt')
)
df_eda = df_eda.loc[:, [el for el in df_eda.columns if el != 'Class']]

standardizer = StandardScaler()
df_eda = pd.DataFrame(
    standardizer.fit_transform(df_eda),
    columns = df_eda.columns
)

outlier_factor = LocalOutlierFactor(
    n_neighbors = 20,
    algorithm = 'ball_tree',
    n_jobs = -1
)
outlier_factor.fit(df_eda)

joblib.dump(
    outlier_factor,
    PROJECT_PATH.joinpath('outputs', 'eda', 'local-outlier-factor.joblib')
)