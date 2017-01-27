import os
import time
import json
from urllib.request import urlopen
from urllib.parse import urljoin
from urllib.error import HTTPError

from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm


def save_and_reload_df(func):
    """
    Decorator that saves the dataframe computed by the function
    and loads it if it was already saved
    """
    def func_wrapper(*args, overwrite=False, **kwargs):
        csv_path = "data/{}.csv".format(func.__name__)
        if not os.path.exists(csv_path) or overwrite:
            df = func(*args, **kwargs)
            df.to_csv(csv_path)
        else:
            print("Reading dataframe from {}".format(csv_path))
            df = pd.read_csv(csv_path, index_col=0)
        return df
    return func_wrapper


def extract_spec(spec):
    key = spec.find("th", {"scope": "row"})
    if key:
        key = key.text
        key = key.replace("\n", " ").strip()
        value = spec.find("td").text
        value = value.replace("\n", " ")
        value = value.replace(u'\xa0', u' ')
        value = value.replace('\t', ' ').strip()
    else:
        return None, None
    return key, value


def get_specs(url):
    """Return specs as a dictionary"""
    html_doc = urlopen(url, timeout=5)
    html_doc = html_doc.read()
    soup = BeautifulSoup(html_doc, "html.parser")

    specs = {}
    soup_price = soup.find("div", {"id": "results"})
    soup_price = soup_price.find("div", {"class": "panel"})
    try:
        specs["prix"] = soup_price.find("a", {"class": "go"}).text
    except AttributeError:
        specs["prix"] = None
    soup = soup.find("div", {"id": "specs"})
    for spec in soup.find_all("tr"):
        key, value = extract_spec(spec)
        if key:
            specs[key] = value

    return specs


def get_laptop_urls_in_page(page_url):
    root_url = "http://www.comparez-malin.fr/informatique/pc-portable/"
    html_doc = urlopen(page_url).read()
    soup = BeautifulSoup(html_doc, "html.parser")
    laptop_blocks = soup.find_all("div", {"class": "product"})
    specs_urls = {}
    for block in laptop_blocks:
        try:
            key = block["id"]
            url = block.find("a", {"class": "white"})["href"]
            if "tablette" not in url:
                url = urljoin(root_url, url.split('/')[-1])
                specs_urls[key] = url
        except KeyError:
            pass
    return specs_urls


def add_columns(df, columns):
    """Add columns to a dataframe"""
    # Remove columns that are already there
    columns = set(columns) - set(df.columns)
    df_columns = pd.DataFrame(columns=columns)
    df = df.join(df_columns, how='outer')
    return df


@save_and_reload_df
def get_laptops_urls():
    """Get links to each laptop page in a dataframe"""
    root_url = "http://www.comparez-malin.fr/informatique/pc-portable/{}"
    n = 265
    specs_urls = {}
    for i in tqdm(range(n)):
        page_url = root_url.format(i+1)
        specs_urls.update(get_laptop_urls_in_page(page_url))

    # Convert urls to dataframe
    s = pd.Series(specs_urls, name='url')
    df = s.to_frame()
    df.to_csv(csv_path)
    return df


@save_and_reload_df
def get_all_laptops_specs(df_laptops_urls, overwrite=False):
    """Get specs for all laptops urls"""
    df = df_laptops_urls
    # Initialize columns
    url = df.iloc[0]["url"]
    specs = get_specs(url)
    columns = set(specs.keys())
    df = add_columns(df, columns)
    columns = set(df.columns)

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row.isnull().values[1:].all():
            url = row["url"]
            specs = get_specs(url)
            if len(specs) == 0:
                print(url)
                pass
            specs["url"] = url
            new_cols = set(specs.keys())
            if (new_cols != columns):
                df = add_columns(df, new_cols - columns)
                columns = set(df.columns)
            df.loc[index] = specs
    return df


def get_cpu_benchmark(cpu_name):
    root_url = "http://www.cpubenchmark.net/cpu.php?cpu={}"
    try:
        url = root_url.format(cpu_name.replace(" ", "+"))
        html_doc = urlopen(url).read()
        soup = BeautifulSoup(html_doc, "html.parser")
        # Square with perf and single thread rating
        soup = soup.find("td", {"style": "text-align: center"})
        benchmark = int(soup.find("span").text)
    except HTTPError:
        benchmark = None
    return benchmark


def get_gpu_benchmark(gpu_name):
    root_url = "http://www.videocardbenchmark.net/gpu.php?gpu={}"
    try:
        url = root_url.format(gpu_name.replace(" ", "+"))
        html_doc = urlopen(url).read()
        soup = BeautifulSoup(html_doc, "html.parser")
        # Square with perf and single thread rating
        soup = soup.find_all("td", {"style": "text-align: center"})[-1]
        benchmark = int(soup.find("span").text)
    except (HTTPError, IndexError):
        benchmark = None
    return benchmark


@save_and_reload_df
def get_cpu_dataframe(cpus):
    """Take a list of cpus (strings) and get there benchmark"""
    df_cpu = pd.DataFrame(cpus, columns=["processeur"])
    for index, row in tqdm(df_cpu.iterrows(), total=df_cpu.shape[0]):
        benchmark = get_cpu_benchmark(row["processeur"])
        df_cpu.loc[index, "cpu_benchmark"] = benchmark
    return df_cpu


@save_and_reload_df
def get_gpu_dataframe(gpus):
    """Take a list of gpus (strings) and get there benchmark"""
    df_gpu = pd.DataFrame(gpus, columns=["puce_graphique_dédiée"])
    for index, row in tqdm(df_gpu.iterrows(), total=df_gpu.shape[0]):
        gpu_name = row["puce_graphique_dédiée"]
        gpu_name = gpu_name.replace("GT ", "").replace("\t(SLI)", "")
        benchmark = get_gpu_benchmark(gpu_name)
        if benchmark is None:
            benchmark = 0
        df_gpu.loc[index, "gpu_benchmark"] = benchmark
    return df_gpu
