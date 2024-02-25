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

distance_network = joblib.load(
    PROJECT_PATH.joinpath('outputs/eda/30NeighboursTransformer.joblib')
)

outlier_factor = LocalOutlierFactor(
    n_neighbors = 20,
    metric = 'precomputed',
    n_jobs = -1
)
outlier_factor.fit(distance_network)

joblib.dump(
    outlier_factor,
    PROJECT_PATH.joinpath('outputs', 'eda', 'local-outlier-factor.joblib')
)