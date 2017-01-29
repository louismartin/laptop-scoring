
import json
from multiprocessing.pool import ThreadPool
import os
from queue import Queue
import sys
from threading import Thread
import time
from urllib.request import urlopen
from urllib.parse import urljoin
from urllib.error import HTTPError

from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

root_url = "http://www.comparez-malin.fr/informatique/pc-portable/"


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


def url2soup(url):
    """Fetch a webpage a return a BeautifulSoup soup object from its HTML"""
    try:
        html_handler = urlopen(url)
        html = html_handler.read()
        soup = BeautifulSoup(html, "html.parser")
    except HTTPError as e:
        print("Error fetching {}: {}".format(url, e))
        soup = None
    return soup


def get_laptop_urls_in_page(page_url):
    """Get all links to laptops specs in one page"""
    soup = url2soup(page_url)
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
            # Ads don't have an id so we just pass
            pass
    return specs_urls


def add_columns(df, columns):
    """Add columns to a dataframe"""
    # Remove columns that are already there
    columns = set(columns) - set(df.columns)
    df_columns = pd.DataFrame(columns=columns)
    df = df.join(df_columns, how='outer')
    return df


def get_max_page():
    """Get maximum page of laptops"""
    soup = url2soup(root_url)
    # Find arrow at the bottom pointing to last page
    # There are 2 arrows, get the last
    arrow = soup.find_all("a", {"aria-label": "Next"})
    max_page = int(arrow[-1]["rel"][0])
    return max_page


@save_and_reload_df
def get_laptops_urls(n_threads=16):
    """Get links to each laptop page in a dataframe"""
    max_page = get_max_page()
    page_urls = [urljoin(root_url, "{}").format(i+1) for i in range(max_page)]

    specs_urls = {}
    # Parallel retrieval
    results = ThreadPool(n_threads).imap_unordered(
        get_laptop_urls_in_page, page_urls
    )
    for laptop_urls in tqdm(results, total=len(page_urls)):
        specs_urls.update(laptop_urls)

    # Convert urls to dataframe
    s = pd.Series(specs_urls, name='url')
    df = s.to_frame()
    return df


# Specifications extraction
def extract_spec(spec):
    """Extract a single spec from a BeautifulSoup object"""
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
    soup = url2soup(url)

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


@save_and_reload_df
def get_all_laptops_specs(df_laptops_urls, n_threads=16):
    """Get specs for all laptops urls"""
    df = df_laptops_urls

    # Initialize columns
    url = df.iloc[0]["url"]
    specs = get_specs(url)
    columns = set(specs.keys())
    df = add_columns(df, columns)
    columns = set(df.columns)

    def fetch_and_process():
        while True:
            index, row = q.get()
            if row.isnull().values[1:].all():
                url = row["url"]
                specs = get_specs(url)
                if len(specs) == 0:
                    print(url)
                    pass
                specs["url"] = url
                df.loc[index] = specs
            q.task_done()

    q = Queue(n_threads * 2)
    for i in range(n_threads):
        t = Thread(target=fetch_and_process)
        t.daemon = True
        t.start()
    try:
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            q.put((index, row))
        start_time = time.time()
        q.join()
        print(time.time() - start_time)
    except KeyboardInterrupt:
        sys.exit(1)

    return df


def get_cpu_benchmark(cpu_name):
    root_url = "http://www.cpubenchmark.net/cpu.php?cpu={}"
    url = root_url.format(cpu_name.replace(" ", "+"))
    soup = url2soup(url)
    if soup:
        # Square with perf and single thread rating
        soup = soup.find("td", {"style": "text-align: center"})
        benchmark = int(soup.find("span").text)
    else:
        benchmark = None
    return benchmark


def get_gpu_benchmark(gpu_name):
    root_url = "http://www.videocardbenchmark.net/gpu.php?gpu={}"
    try:
        sli = "(SLI)" in gpu_name
        pascal_models = ["1050", "1060", "1070", "1080"]
        pascal = any([model in gpu_name for model in pascal_models])
        gpu_name = gpu_name.replace(" (SLI)", "")
        url = root_url.format(gpu_name.replace(" ", "+"))
        soup = url2soup(url)
        if soup:
            # Square with perf and single thread rating
            soup = soup.find_all("td", {"style": "text-align: center"})[-1]
            benchmark = int(soup.find("span").text)
            # SLI are slightly better
            benchmark *= (1 + 0.2*sli)
            # The benchmark for pascal GPUs is the desktop benchmark !
            benchmark *= (1 - 0.2*pascal)
            benchmark = int(benchmark)
        else:
            benchmark = None
    except IndexError:
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
