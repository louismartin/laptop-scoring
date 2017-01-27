from urllib.request import urlopen

import numpy as np
from bs4 import BeautifulSoup
from IPython.core.display import display, HTML


def scale(col):
    """Scale a pandas dataframe column (0 mean and 1 variance)"""
    col = (col-col.mean())/col.std()
    return col


def ease(x):
    """Ease extreme values"""
    return np.sign(x)*np.sqrt(abs(x))


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
        # Display html
        display(HTML(str(soup)))
