"""
Provide example use case of raincloud plot.
"""
### Imports
from pathlib import Path
import dotenv
from scipy.stats import norm, lognorm
from numpy import append, reshape
import matplotlib.pyplot as plt
from pandas import DataFrame
from credit_fraud.raincloud import pyplot_raincloud

PROJECT_PATH = Path(dotenv.find_dotenv()).absolute().parent

if __name__ == '__main__':

    x = append(
        norm.rvs(loc = 5, scale = 2, size = 200),
        lognorm.rvs(s = 0.7, loc = 7, size = 200)
    )

    x = DataFrame(reshape(x, (-1, 1)), columns = ['x'])

    fig, ax = plt.subplots()

    cloud = pyplot_raincloud(
        x,
        'x',
        ax
    )

    ax.set_title('Example Raincloud')
    
    fig.savefig(
        PROJECT_PATH.joinpath('test', 'test_outputs', 'raincloud_example.png')
    )