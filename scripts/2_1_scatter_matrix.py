"""
Generates visuals of single variable distributions wrt credit fraud
presence
"""
### Imports
from pathlib import Path
from dotenv import find_dotenv
from operator import mul
from itertools import starmap
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.axes import Axes
from sklearn.metrics import PrecisionRecallDisplay

### Constants
PROJECT_PATH = Path(find_dotenv('.gitignore')).absolute().parent

### Functions
def set_ylabel(ax: Axes, label: str, **kwargs) -> None:
    """
    Calls set_ylabel method on an matplotlib Axes object.
    """
    ax.set_ylabel(label, **kwargs)

########################################################################
########################################################################

if __name__ == '__main__':

    df_eda = pd.read_parquet(
        PROJECT_PATH.joinpath('data', 'eda-pack', 'df_eda.pqt')
    )

    scales = df_eda['Class'].value_counts(normalize = True).to_dict()

    classes = df_eda.groupby('Class')

    feature_cols = [el for el in df_eda.columns if el != 'Class']

    n_rows: int = len(feature_cols)
    n_cols: int = 3

    figsize = starmap(mul, zip((n_cols, n_rows), rcParams['figure.figsize']))

    fig, axes = plt.subplots(
        nrows = n_rows,
        ncols = 3,
        figsize = [*figsize]
    )

    _ = tuple(
        starmap(
            partial(set_ylabel, fontsize = 12),
            zip(axes[:, 0], feature_cols)
        )
    )

    plots: dict = dict()

    for i, col in enumerate(feature_cols):

        for clsname, frame in iter(classes):

            subtitle = 'Scaled Violin'
            axes[i, 0].set_title(subtitle)

            plots[f'{subtitle}_{col}_{clsname}'] = axes[i, 0].violinplot(
                frame[col],
                vert = False,
                showmeans = False,
                showmedians = False,
                showextrema = False
            )
            # scaling violin bodies
            for body in plots[f'{subtitle}_{col}_{clsname}']['bodies']:
                # get width values for violin bodies
                widths = body.get_paths()[0].vertices[:, 1] - 1
                widths *= scales[clsname]
                body.get_paths()[0].vertices[:, 1] = widths + 1 # Update body

            subtitle = 'Unscaled Violin'
            axes[i, 1].set_title(subtitle)

            plots[f'{subtitle}_{col}_{clsname}'] = axes[i, 1].violinplot(
                frame[col],
                vert = False,
                showmeans = False,
                showmedians = False,
                showextrema = False
            )

        subtitle = 'Precision-Recall using feature as threshold'
        plots[f'{subtitle}_{col}_PR_less_than'] =\
            PrecisionRecallDisplay.from_predictions(
                df_eda['Class'],
                df_eda[col],
                pos_label = 1,
                ax = axes[i, 2],
                label = 'GE - Threshold'
            )
        plots[f'{subtitle}_{col}_PR_greater_than'] =\
            PrecisionRecallDisplay.from_predictions(
                df_eda['Class'],
                -df_eda[col],
                pos_label = 1,
                ax = axes[i, 2],
                label = 'LE - Threshold'
            )
        axes[i, 2].set_title(subtitle)
