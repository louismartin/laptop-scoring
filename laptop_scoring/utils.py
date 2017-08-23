import os
import time
from urllib.request import urlopen
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from IPython.core.display import display, HTML, clear_output
from ipywidgets import widgets
import numpy as np
import pandas as pd
import dominate
from dominate.tags import *


SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, "data")
BACKUP_DIR = os.path.join(DATA_DIR, "backup")
IMG_DIR = os.path.join(DATA_DIR, "images")
# Make sure the directories exist
for directory in [DATA_DIR, BACKUP_DIR, IMG_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)


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

            # Back up the old file if it exists
            if os.path.exists(csv_path) and overwrite:
                timestamp = int(time.time())
                backup_filename = "{}_{}.csv".format(func.__name__, timestamp)
                backup_path = os.path.join(BACKUP_DIR, backup_filename)
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


def row2html(row):
    row_html = div(cls="row")
    with row_html:
        hr()
        with div(cls="col-xs-12 col-sm-12 col-md-12 col-lg-4"):
            img(src=row["image_url"], style="width:290px;")
        with div(cls="col-xs-12 col-sm-12 col-md-6 col-lg-5"):
            title = "{} {}".format(row["marque"], row["référence"])
            h1(title, style="margin-bottom:10px")
            with ul():
                li('Écran de {}" {}{}'.format(
                    row["taille"],
                    row["type_de_dalle"],
                    " (anti-reflets)" if (row["anti-reflet"] == "oui") else ""
                  ))
                li('Résolution {}'.format(row["résolution"]))
                formatting_string = 'Processeur {} ({} x {min_freq:.1f} à \
                                     {max_freq:.1f} GHz) - ({bench})'
                li(formatting_string.format(
                    row["processeur"], int(row["coeurs"]),
                    min_freq=row["min_freq"], max_freq=row["max_freq"],
                    bench=int(row["cpu_benchmark"])
                  ))
                li('Carte graphique: {} - ({})'.format(
                    row["puce_graphique_dédiée"], int(row["gpu_benchmark"])
                  ))
                li('Ram: {} Go'.format(int(row["mémoire_ram"])))
                li(row["disque_dur"])
                li(a("Voir les caractéristiques complètes", href=row["url"]))
        with div(cls="col-xs-12 col-sm-12 col-md-6 col-lg-3",
                 style="margin-top:80px"):
            with a(cls="btn btn-lg btn-block btn-success",
                   href=row["url"], target="_blank"):
                strong("{} €".format(row["prix"]))
                br()
                span(" (min {} €)".format(row["prix_min"]))
        return row_html


def display_laptop(row, offline=False, full=False):
    """Display summary of each laptop in a notebook"""
    url = row["url"]
    if offline:
        row_html = row2html(row)
        display(HTML(str(row_html)))
    elif full:
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
    print("total: {0:.2f}".format(row["total"]))
    row = row.drop(["total", "score"])
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


def draw_rank_page(df, start=0, offline=False, df_score=None):
    n = 5
    end = start + n

    def on_button_clicked(b):
        container.close()
        clear_output()
        draw_rank_page(df, start=b.value, offline=offline, df_score=df_score)

    btn_next = widgets.Button(description="Next", value=(start+n))
    btn_next.on_click(on_button_clicked)
    btn_prev = widgets.Button(description="Previous", value=(start-n))
    btn_prev.on_click(on_button_clicked)
    container = widgets.VBox(children=[btn_next, btn_prev])
    display(container)
    print("Page {}.".format((start//n) + 1))
    for i, (index, row) in enumerate(df.iloc[start:end].iterrows()):
        display_laptop(row, offline=offline)
        if df_score is not None:
            print_score(df_score.loc[index])