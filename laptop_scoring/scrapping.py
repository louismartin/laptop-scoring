from queue import Queue
import os
import re
import sys
from threading import Thread
import time
from urllib.request import urlopen, urlretrieve
from urllib.parse import urljoin
from urllib.error import HTTPError, URLError

from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from tqdm import tqdm

from laptop_scoring.utils import save_and_reload_df, IMG_DIR


# Global variables
ROOT_URL = "http://www.comparez-malin.fr/informatique/pc-portable/"


def url2soup(url):
    """Fetch a webpage a return a BeautifulSoup soup object from its HTML"""
    try:
        html_handler = urlopen(url)
        html = html_handler.read()
        soup = BeautifulSoup(html, "html.parser")
    except (HTTPError, URLError, ConnectionResetError) as e:
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
                    url = urljoin(ROOT_URL, url.split('/')[-1])
                    price = get_price(block)
                    image_soup = block.find("div", {"class": "gallery"})
                    if image_soup:
                        image_url = image_soup.a["href"]
                    else:
                        image_url = None
                    specs_urls[key] = (url, image_url, price)
    else:
        specs_urls = None
    return specs_urls


def add_columns(df, columns):
    """Add columns to a dataframe"""
    # Take only columns that were not already there
    new_columns = set(columns) - set(df.columns)
    if len(new_columns) == 0:
        return df
    df_new_columns = pd.DataFrame(columns=new_columns)
    df = df.join(df_new_columns, how='outer')
    return df


def add_columns_inplace(df, columns):
    """Add columns to a dataframe INPLACE"""
    # Take only columns that were not already there
    new_columns = set(columns) - set(df.columns)
    for col in new_columns:
        df[col] = None


def get_max_page():
    """Get maximum page of laptops"""
    soup = url2soup(ROOT_URL)
    # Find arrow at the bottom pointing to last page
    # There are 2 arrows, get the last
    arrow = soup.find_all("a", {"aria-label": "Next"})
    max_page = int(arrow[-1]["rel"][0])
    return max_page


@save_and_reload_df
def get_laptops_urls(n_threads=16, max_page=np.inf):
    """Get links to each laptop page in a dataframe."""
    max_page = min(max_page, get_max_page())
    page_urls = [urljoin(ROOT_URL, "{}").format(i+1) for i in range(max_page)]

    # Parallel retrieval
    specs_urls = {}

    def fetch_and_process():
        """Process what's in the queue until there is nothing left"""
        while True:
            url = q.get()
            laptop_urls = get_laptop_urls_in_page(url)
            if laptop_urls is not None:
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
            url = urljoin(ROOT_URL, "{}").format(i+1)
            q.put(url)
        # Wait for everybody to finish
        q.join()
    except KeyboardInterrupt:
        sys.exit(1)

    # Convert urls to dataframe
    df = pd.DataFrame(specs_urls).transpose()
    df.columns = ["url", "image_url", "prix"]
    return df


def save_images(df_urls):
    df_urls["image_url"] = df_urls["image_url"].fillna("")
    for index, row in tqdm(df_urls.iterrows(), total=df_urls.shape[0]):
        image_url = row["image_url"]
        if len(image_url):
            filename = "{}.jpg".format(index)
            path = os.path.join(IMG_DIR, filename)
            if not os.path.exists(path):
                urlretrieve(image_url, path)


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
def get_all_laptops_specs(df_urls, n_threads=16):
    """Get specs for all laptops urls"""
    df = df_urls.copy()

    # Initialize columns
    url = df.iloc[0]["url"]
    specs = get_specs(url)
    columns = set(list(specs.keys()) + list(df_urls.columns))
    df = add_columns(df, columns)

    def fetch_and_process():
        """Process what's in the queue until there is nothing left"""
        while True:
            index, row = q.get()
            specs = get_specs(row["url"])
            # Add columns that could be missing
            add_columns_inplace(df, specs.keys())
            if specs:
                # Add values that were already there in df_urls
                for col in df_urls.columns:
                    specs[col] = row[col]
                df.loc[index, list(specs.keys())] = list(specs.values())
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
    benchmark = 0
    if soup:
        # Square with perf and single thread rating
        soup = soup.find("td", {"style": "text-align: center"})
        if soup:
            benchmark = int(soup.find("span").text)
    return benchmark


def get_gpu_benchmark(gpu_name):
    root_url = "http://www.videocardbenchmark.net/gpu.php?gpu={}"
    try:
        sli = "(SLI)" in gpu_name
        pascal_models = ["1050", "1060", "1070", "1080"]
        pascal = any([model in gpu_name for model in pascal_models])
        gpu_name = gpu_name.replace(" (SLI)", "")
        gpu_name = re.sub("\d Go de mémoire vive", "", gpu_name)
        url = root_url.format(gpu_name.replace(" ", "+"))
        soup = url2soup(url)
        if soup:
            # Square with perf and single thread rating
            soup = soup.find_all("td", {"style": "text-align: center"})[-1]
            benchmark = int(soup.find("span").text)
            # SLI are slightly better
            benchmark *= (1 + 0.2*sli)
            # The benchmark for pascal GPUs is the desktop benchmark !
            benchmark *= (1 - 0.1*pascal)
            benchmark = int(benchmark)
        else:
            benchmark = 0
    except IndexError:
        benchmark = 0
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
        gpu_name = gpu_name.replace("GT ", "").replace("\t", "")
        benchmark = get_gpu_benchmark(gpu_name)
        df_gpu.loc[index, "gpu_benchmark"] = benchmark
    return df_gpu
