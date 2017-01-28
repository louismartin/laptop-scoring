from urllib.request import urlopen
from urllib.parse import urljoin

import numpy as np
from bs4 import BeautifulSoup
from IPython.core.display import display, HTML


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
