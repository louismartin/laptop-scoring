import glob
import os

import numpy as np
import pandas as pd

from laptop_scoring.utils import BACKUP_DIR


def get_min_price(df_urls):
    """
    returns the all time minimum price of current laptops based on dataframes
    stored in the backup directory
    """
    paths = glob.glob(os.path.join(BACKUP_DIR, "get_laptops_urls*.csv"))
    cols = []
    for i, path in enumerate(paths):
        timestamp = int(path.split("_")[-1].split(".")[0])
        df_temp = pd.read_csv(path, index_col=0)
        suffix = "_{}".format(timestamp)
        df_urls = df_urls.join(df_temp, how="left", rsuffix=suffix)
        cols.append("prix" + suffix)

    return df_urls[cols].min(axis=1)


def process_and_clean(df):
    """RIP good practices, my laziness won over you"""
    cols_str = ["stockage", "usb", "composition"]
    df[cols_str] = df[cols_str].fillna("")
    df["prix_public"] = df["prix_public"].str.strip("€").str.replace(" ", "")\
        .str.replace(",", ".").astype(float)
    df[["coeurs", "min_freq", "max_freq"]] = df["fréquence"]\
        .str.split(expand=True)[[0, 2, 4]].astype(float)
    df["single_core_benchmark"] = df["cpu_benchmark"] / df["coeurs"]
    df["pdt_max"] = df["pdt_max"].str.split(expand=True)[0].astype(float)
    df["mémoire_ram"] = df["mémoire_ram"].str.split(expand=True)[0]\
        .astype(float)

    # Split stockage in sshd (bool), hdd_size, hdd_speed, ssd_size
    df["sshd"] = df["stockage"].apply(lambda x: ("cache SSD" in x))
    df[['hdd_size', 'hdd_speed']] = df['stockage'].str.extract('(\d+) GoHDD\d\.\d"(\d+) tr/min', expand=True).fillna(0).astype(int)
    df['ssd_size'] = df['stockage'].str.extract('(\d+) GoSSD', expand=True).fillna(0).astype(int)
    df["res_width"] = df["résolution"].str.split(expand=True)[0].astype(float)
    df["taille"] = df["taille"].str.split('" ', expand=True)[0].astype(float)
    df[["width", "depth", "height"]] = df["dimensions"]\
        .str.split(expand=True)[[0, 2, 4]].astype(float)
    df[["width", "depth", "height"]] = df["dimensions"]\
        .str.split("x", expand=True)
    # Sometimes heights are written "17 - 18" so take the max
    df["height"] = df["height"].str.replace(" mm", "")\
        .str.split("-", expand=True).fillna(0).astype(float).max(axis=1)
    df[["width", "depth"]] = df[["width", "depth"]].astype(float)

    # Screen to body ratio
    ratio_width_diag = 16 / np.sqrt(16**2 + 9**2)
    ratio_mm_inch = 25.4
    coef = ratio_width_diag * ratio_mm_inch
    df["screen_size_h"] = df["taille"].apply(lambda x: coef*x)
    df["screen_size_v"] = 9/16 * df["screen_size_h"]
    df["screen_to_body"] = df.apply(
        lambda row: (row["screen_size_h"]*row["screen_size_v"]) /
                    (row["width"]*row["depth"]),
        axis=1
        )

    df["poids"] = df["poids"].str.replace("kg", "").str.replace("g", "")\
        .str.strip().astype(float)
    # Convert weights expressed in grams to kilograms
    df["poids"] = df["poids"].apply(lambda x: x if (x < 100) else x/1000)
    df["type_c"] = df["usb"].apply(
        lambda x: x.split("Type-C")[-1] if "Type-C" in x else ""
        )
    df[["day", "month", "year"]] = df["date"].str.split("/", expand=True)\
        .astype(float)

    # Clean rows for easier reading
    df["processeur"] = df["processeur"].str.replace("Intel Core", "")
    df["puce_graphique_dédiée"] = df["puce_graphique_dédiée"]\
        .str.replace("Nvidia", "").str.replace("GeForce", "")\
        .str.replace("GTX", "").str.strip()

    return df
