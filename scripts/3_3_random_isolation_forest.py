"""
Create Random Isolation Forest model and get isolation scores for credit
fraud data.
"""
### Imports
import pathlib as pl
import dotenv
from numpy import save as npz_save
from pandas import read_parquet as pd_read_pqt, DataFrame as PandasDataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib

### Constants
PROJECT_PATH = pl.Path(dotenv.find_dotenv()).absolute().parent

df_eda = pd_read_pqt(PROJECT_PATH.joinpath('data', 'eda-pack', 'df_eda.pqt'))
df_eda = df_eda.loc[:, [el for el in df_eda.columns if el != 'Class']]

standardizer = StandardScaler()
df_eda = PandasDataFrame(
    standardizer.fit_transform(df_eda),
    columns = df_eda.columns
)

iso_forest = IsolationForest(
    n_estimators = 1000,
    n_jobs = -1,
    random_state = 19
)
iso_forest = iso_forest.fit(df_eda)
iso_scores = iso_forest.decision_function(df_eda)

# save
joblib.dump(
    iso_forest,
    PROJECT_PATH.joinpath('outputs', 'eda', 'isolation-forest.joblib')
)

with open(PROJECT_PATH.joinpath('outputs/eda/iso_scores.npz'), 'wb') as f:
    npz_save(f, iso_scores)
