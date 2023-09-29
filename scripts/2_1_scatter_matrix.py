"""
Generates a scatter matrix of df_eda using plotly.
"""
### Imports
import pandas as pd
import dotenv
import pathlib as pl
import plotly.express as px

### Constants
PROJECT_PATH = pl.Path(dotenv.find_dotenv()).absolute().parent
DEFAULT_HEIGHT = 400
DEFAULT_WIDTH = 600

########################################################################
########################################################################

if __name__ == '__main__':

    df_eda = pd.read_parquet(
        PROJECT_PATH.joinpath('data', 'eda-pack', 'low-res.pqt')
    )

    fig = px.scatter_matrix(
        df_eda,
        (dims := [el for el in df_eda.columns if el != 'Class']),
        'Class',
        width = 10 * DEFAULT_WIDTH,
        height = 10 * DEFAULT_HEIGHT
    )

    fig.write_html(
        PROJECT_PATH.joinpath('outputs', 'eda', 'scatter_matrix.html')
    )