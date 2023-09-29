"""
Applied kmeans to 0-labelled records, reducing ammount of data to plot.
"""
### Imports
import pathlib as pl
import dotenv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

### Constants
PROJECT_PATH = pl.Path(dotenv.find_dotenv()).absolute().parent

df = pd.read_parquet(PROJECT_PATH.joinpath('data', 'eda-pack', 'df_eda.pqt'))

y_col = 'Class'
feature_cols =[el for el in df.columns if el != y_col]

df_scaled = df.copy()
standardizer = StandardScaler()
X = standardizer.fit_transform((X := df_scaled[feature_cols]))

df_scaled_0 = df_scaled.loc[df_scaled['Class'] == 0]
df_scaled_1 = df_scaled.loc[df_scaled['Class'] == 1]

clusterer = KMeans(n_clusters = 2000)
clusterer.fit(df_scaled_0[feature_cols])

joblib.dump(
    clusterer,
    PROJECT_PATH.joinpath('outputs', 'eda', 'kmeans-2000.joblib')
)

low_res = pd.DataFrame(clusterer.cluster_centers_, columns = feature_cols)
low_res['Class'] = 0
low_res = pd.concat([low_res, df_scaled_1], ignore_index = True)

low_res.to_parquet(PROJECT_PATH.joinpath('data', 'eda-pack', 'low-res.pqt'))
