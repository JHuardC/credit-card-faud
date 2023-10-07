"""
This script splits the creditcard dataset into train test datasets and
an Exploratory Data Analysis dataset.
"""
### Imports
import pathlib as pl
import dotenv
from numpy import save as numpy_save
import joblib
from pandas import DataFrame, Series, read_json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

### Constants
PROJECT_PATH = pl.Path(dotenv.find_dotenv()).absolute().parent
TRAIN_TEST_PATH = PROJECT_PATH.joinpath('data', 'train-test-pack')
EDA_PATH = PROJECT_PATH.joinpath('data', 'eda-pack')

########################################################################
########################################################################

if __name__ == '__main__':

    df = read_json(
        PROJECT_PATH.joinpath('data', 'data', 'creditcard_json.json')
    )

    # encode y from text to int
    y = df.pop('Class')
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        df, y,
        test_size = 0.2,
        random_state = 71,
        stratify = y
    )

    print(
        Series(y_train, name = 'y_train').value_counts(normalize = True),
        Series(y_test, name = 'y_test').value_counts(normalize = True),
        sep = '\n\n---\n\n'
    )
    
    # create eda dataframe
    df_eda = DataFrame(X_train, columns = df.columns)
    df_eda['Class'] = y_train

    # Saving outputs
    joblib.dump(le, TRAIN_TEST_PATH.joinpath('lbl-enc.joblib'))

    with open(TRAIN_TEST_PATH.joinpath('X_train.npm'), 'wb') as f:
        numpy_save(f, X_train)

    with open(TRAIN_TEST_PATH.joinpath('y_train.npm'), 'wb') as f:
        numpy_save(f, y_train)

    with open(TRAIN_TEST_PATH.joinpath('X_test.npm'), 'wb') as f:
        numpy_save(f, X_test)

    with open(TRAIN_TEST_PATH.joinpath('y_test.npm'), 'wb') as f:
        numpy_save(f, y_test)
    
    df_eda.to_parquet(EDA_PATH.joinpath('df_eda.pqt'))
