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
    and loads it if it was already saved.
    """
    def func_wrapper(*args, overwrite=False, **kwargs):
        # Create all the paths and filenames necessary
        data_dir = "data"
        filename = "{}.csv".format(func.__name__)
        csv_path = os.path.join("data", filename)
        # The file already exists so we just read it from disk
        if os.path.exists(csv_path) and not overwrite:
            print("Reading dataframe from {}".format(csv_path))
            df = pd.read_csv(csv_path, index_col=0)
        # Either the file does not exist or we want to compute it again
        else:
            # Make sure the data directory already exists
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            # Compute the new file
            df = func(*args, **kwargs)

            # Back up the old file if it exists
            if os.path.exists(csv_path) and overwrite:
                timestamp = int(time.time())
                backup_filename = "{}_{}.csv".format(func.__name__, timestamp)
                backup_dir = os.path.join(data_dir, "backup")
                backup_path = os.path.join(backup_dir, backup_filename)
                if not os.path.exists(backup_dir):
                    os.makedirs(backup_dir)
                os.rename(csv_path, backup_path)

            # Save the new file
            df.to_csv(csv_path)
        return df
    return func_wrapper


def url2soup(url):
    """Fetch a webpage a return a BeautifulSoup soup object from its HTML"""
    try:
        html_handler = urlopen(url)
        html = html_handler.read()
        soup = BeautifulSoup(html, "html.parser")
    except HTTPError as e:
        print("Error fetching {} : {}".format(url.lower(), e))
        soup = None
    return soup


def get_price(soup):
    """Retrieve price inside a button from a BeautifulSoup object"""
    tag = soup.find("a", {"class": "go"})
    if tag:
        price = tag.text
        price = float(price.strip("€").replace(" ", "").replace(",", "."))
    else:
        # When the button with the price is not found, laptop is unavailable
        price = None
    return price


def get_laptop_urls_in_page(page_url):
    """Get all links to laptops specs adn prices in one page"""
    soup = url2soup(page_url)
    if soup:
        laptop_blocks = soup.find_all("div", {"class": "product"})
        specs_urls = {}
        for block in laptop_blocks:
            if block.has_attr("id"):
                key = block["id"]
                url = block.find("a", {"class": "white"})["href"]
                if "tablette" not in url:
                    url = urljoin(root_url, url.split('/')[-1])
                    price = get_price(block)
                    specs_urls[key] = (url, price)
    else:
        specs_urls = None
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

    # Parallel retrieval
    specs_urls = {}

    def fetch_and_process():
        """Process what's in the queue until there is nothing left"""
        while True:
            url = q.get()
            laptop_urls = get_laptop_urls_in_page(url)
            if laptop_urls:
                specs_urls.update(laptop_urls)
            else:
                # The request did not succeed, do it again
                q.put(url)
            q.task_done()

    q = Queue(n_threads * 2)
    for i in range(n_threads):
        t = Thread(target=fetch_and_process)
        t.daemon = True
        t.start()
    try:
        # Add arguments
        for i in tqdm(range(max_page)):
            url = urljoin(root_url, "{}").format(i+1)
            q.put(url)
        # Wait for everybody to finish
        q.join()
    except KeyboardInterrupt:
        sys.exit(1)

    # Convert urls to dataframe
    df = pd.DataFrame(specs_urls).transpose()
    df.columns = ["url", "prix"]
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
    if soup:
        specs = {}
        soup = soup.find("div", {"id": "specs"})
        for spec in soup.find_all("tr"):
            key, value = extract_spec(spec)
            if key:
                specs[key] = value
    else:
        specs = None
    return specs


@save_and_reload_df
def get_all_laptops_specs(df_laptops_urls, n_threads=16):
    """Get specs for all laptops urls"""
    df = df_laptops_urls.copy()

    # Initialize columns
    url = df.iloc[0]["url"]
    specs = get_specs(url)
    columns = set(specs.keys() + ["url", "prix"])
    df = add_columns(df, columns)
    columns = set(df.columns)

    def fetch_and_process():
        """Process what's in the queue until there is nothing left"""
        while True:
            index, row = q.get()
            specs = get_specs(row["url"])
            if specs:
                specs["url"] = row["url"]
                specs["prix"] = row["prix"]
                df.loc[index] = specs
            else:
                # The request did not succeed, do it again
                q.put((index, row))
            q.task_done()

    q = Queue(n_threads * 2)
    for i in range(n_threads):
        t = Thread(target=fetch_and_process)
        t.daemon = True
        t.start()
    try:
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            q.put((index, row))
        q.join()
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
