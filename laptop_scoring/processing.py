import numpy as np


def process_and_clean(df):
    """RIP PEP8, my laziness won over you"""
    df["prix_public"] = df["prix_public"].str.strip("€").str.replace(" ", "").str.replace(",", ".").astype(float)
    df["prix"] = df["prix"].str.strip("€").str.replace(" ", "").str.replace(",", ".").astype(float)
    df[["coeurs", "min_freq", "max_freq"]] = df["fréquence"].str.split(expand=True)[[0, 2, 4]].astype(float)
    df["single_core_benchmark"] = df["cpu_benchmark"] / df["coeurs"]
    df["pdt_max"] = df["pdt_max"].str.split(expand=True)[0].astype(int)
    df["mémoire_ram"] = df["mémoire_ram"].str.split(expand=True)[0].astype(int)

    # Split disque_dur in sshd (bool), hdd_size, hdd_speed, ssd_size
    df["sshd"] = df["disque_dur"].apply(lambda x: ("cache SSD" in x))
    df["disque_dur"] = df["disque_dur"].str.replace("cache SSD", "").str.replace("(", "").str.replace(")", "")
    df["hdd_size"] = df["disque_dur"].apply(lambda x: int(x.split("tr/min")[0].split()[0]) if ("tr/min" in x) else 0)
    df["hdd_speed"] = df["disque_dur"].apply(lambda x: int((x.split("tr/min")[0].strip().split()[-1])) if ("tr/min" in x) else 0)
    df["ssd_size"] = df["disque_dur"].apply(lambda x: int(x.split("tr/min")[-1].split("Go SSD")[0].split()[-1].replace("SSD", "")) if ("Go SSD" in x) else 0)

    df["res_width"] = df["résolution"].str.split(expand=True)[0].astype(int)
    df["taille"] = df["taille"].str.split('" ', expand=True)[0].astype(float)
    df[["width", "depth", "height"]] = df["dimensions"].str.split(expand=True)[[0, 2, 4]].astype(float)
    df[["width", "depth", "height"]] = df["dimensions"].str.split("x", expand=True)
    # Sometimes heights are written "17 - 18" so take the max
    df["height"] = df["height"].str.replace(" mm", "").str.split("-", expand=True).fillna(0).astype(float).max(axis=1)
    df[["width", "depth"]] = df[["width", "depth"]].astype(float)

    # Screen to body ratio
    ratio_width_diag = 16 / np.sqrt(16**2 + 9**2)
    ratio_mm_inch = 25.4
    coef = ratio_width_diag * ratio_mm_inch
    df["screen_size_h"] = df["taille"].apply(lambda x: coef*x)
    df["screen_size_v"] = 9/16 * df["screen_size_h"]
    df["screen_to_body"] = df.apply(lambda row: (row["screen_size_h"]*row["screen_size_v"]) / (row["width"]*row["depth"]), axis=1)

    df["poids"] = df["poids"].str.replace("kg", "").str.replace("g", "").str.strip().astype(float)
    # Convert weights expressed in grams to kilograms
    df["poids"] = df["poids"].apply(lambda x: x if x<100 else x/1000)
    df["type_c"] = df["usb"].apply(lambda x: x.split("Type-C")[-1] if "Type-C" in x else "")
    df = df.fillna(0)
    return df
