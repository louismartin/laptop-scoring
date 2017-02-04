import os
import time
from urllib.request import urlopen
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from IPython.core.display import display, HTML
import numpy as np
import pandas as pd


DATA_DIR = "data"
BACKUP_DIR = os.path.join(DATA_DIR, "backup")


def save_and_reload_df(func):
    """
    Decorator that saves the dataframe computed by the function
    and loads it if it was already saved.
    """
    def func_wrapper(*args, overwrite=False, **kwargs):
        # Create all the paths and filenames necessary
        filename = "{}.csv".format(func.__name__)
        csv_path = os.path.join(DATA_DIR, filename)
        # The file already exists so we just read it from disk
        if os.path.exists(csv_path) and not overwrite:
            print("Reading dataframe from {}".format(csv_path))
            df = pd.read_csv(csv_path, index_col=0)
        # Either the file does not exist or we want to compute it again
        else:
            # Compute the new file
            df = func(*args, **kwargs)

            # Make sure the data directory already exists
            if not os.path.exists(DATA_DIR):
                os.makedirs(DATA_DIR)

            # Back up the old file if it exists
            if os.path.exists(csv_path) and overwrite:
                timestamp = int(time.time())
                backup_filename = "{}_{}.csv".format(func.__name__, timestamp)
                backup_path = os.path.join(BACKUP_DIR, backup_filename)
                if not os.path.exists(BACKUP_DIR):
                    os.makedirs(BACKUP_DIR)
                os.rename(csv_path, backup_path)

            # Save the new file
            df.to_csv(csv_path)
        return df
    return func_wrapper


def scale(col):
    """Scale a pandas dataframe column (0 mean and 1 variance)"""
    col = (col-col.mean())/col.std()
    return col


def ease(x, method="sqrt", asymetric=False):
    """Ease extreme values"""
    if method == "sqrt":
        def func(x): return np.sqrt(x)
    elif method == "log":
        def func(x): return np.log(1+x)
    else:
        raise ValueError("Method {} is not valid".format(method))

    if asymetric and x < 0:
        # Penalize negatives
        out = (1-np.exp(-x))
    else:
        out = np.sign(x)*func(abs(x))
    return out


def display_laptop(url, full=False):
    """Display summary of each laptop in a notebook"""
    if full:
        iframe = '<iframe width="100%" height="350" src="{}"></iframe>'
        display(HTML(iframe.format(url)))
    else:
        html_doc = urlopen(url, timeout=5)
        html_doc = html_doc.read()
        soup = BeautifulSoup(html_doc, "html.parser")
        # Find block with summary of laptop
        soup = soup.find("div", {"id": "results"})
        soup = soup.find("div", {"class": "panel"})
        soup = soup.find("div", {"class": "row"})
        # Replace seller image with its name
        img = soup.find_all('img')[-1]
        img.name = "h3"
        img.string = img["alt"]
        img.attrs = None

        # Replace relative link to seller with absolute link
        link = soup.find("a", {"class": "price"})
        if link:
            link.attrs["href"] = urljoin(url, link.attrs["href"])

        # Display html
        display(HTML(str(soup)))


def print_score(row):
    print("total: {0:.2f} - prix: {1:.2f}".format(row["total"], row["prix"]))
    row = row.drop(["total", "score", "prix"])
    row_pos = row[row > 0].sort_values(ascending=False)
    row_neg = row[row < 0].sort_values()

    string = ""
    for index, value in row_pos.iteritems():
        string += "{0}: {1:.2f}".format(index[:5], value)
        if len(string) > 100:
            break
        else:
            string += " / "
    print(string)

    string = ""
    for index, value in row_neg.iteritems():
        string += "{0}: {1:.2f}".format(index[:5], value)
        if len(string) > 100:
            break
        else:
            string += " / "
    print(string)
