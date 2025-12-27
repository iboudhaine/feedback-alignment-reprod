"""
Plotting helpers for CSVs produced by this repo.
These are not required to run the learning algorithms.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_csv(xcol, ycol, csv_path: Path, title: str = ""):
    df = pd.read_csv(csv_path)
    plt.figure()
    plt.plot(df[xcol], df[ycol])
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    if title:
        plt.title(title)
    plt.show()
