"""
Create local outlier factor feature
"""
### Imports
import pathlib as pl
import dotenv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsTransformer
import joblib
from scipy.sparse import save_npz

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

nearest_neighbours = KNeighborsTransformer(
    mode = 'distance',
    n_neighbors = 30,
    algorithm = 'ball_tree',
    n_jobs = -1
)
distance_network = nearest_neighbours.fit_transform(df_eda)

### save distance_network
save_npz(
    PROJECT_PATH.joinpath('outputs/eda/distance_network_30.npz'),
    distance_network
)

### save fitted transformer
joblib.dump(
    nearest_neighbours,
    PROJECT_PATH.joinpath('outputs/eda/30NeighboursTransformer.joblib')
)