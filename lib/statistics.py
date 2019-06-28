"""
Helper functions to handle statistics and plot them.
"""
from pathlib import Path
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML


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

    plt.show(block=False)
    plt.close('all')


def html_video_embedding(statistics: List[Dict]) -> HTML:
    best_episode = np.argmax([statistic['score'] for statistic in statistics])
    video_path = list(Path('./video').glob(f'openaigym.video.*.id.video00000{best_episode}.mp4'))[0]

    return HTML(f"""
            <div align="middle">
            <video controls>
                  <source src="{video_path!s}" type="video/mp4">
            </video></div>
            """)
