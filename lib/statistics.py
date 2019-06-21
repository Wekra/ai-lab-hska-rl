"""
Helper functions to handle statistics and plot them.
"""
from typing import List, Dict
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot(statistics: List[Dict], y_limits: Tuple[int, int] = None) -> None:
    x = [statistic['episode'] for statistic in statistics]
    y = [statistic['score'] for statistic in statistics]

    mean = np.mean(y)

    fig, ax = plt.subplots()
    if y_limits:
        ax.set_ylim(y_limits)

    ax.plot(x, y, label="scores")
    ax.plot(x, [mean] * len(x), label=f'mean {mean}')

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
