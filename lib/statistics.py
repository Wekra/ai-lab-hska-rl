"""
Helper functions to handle statistics and plot them.
"""
import matplotlib.pyplot as plt
from typing import List, Dict

def plot(statistics: List[Dict]) -> None:
    x = [statistic['episode'] for statistic in statistics]
    y = [statistic['score'] for statistic in statistics]

    fig, ax = plt.subplots()
    ax.plot(x, y)
