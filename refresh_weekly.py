from laptop_scoring.processing import get_min_price
from laptop_scoring import scrapping

# Get links to each laptop page in a  dataframe
df_urls = scrapping.get_laptops_urls(overwrite=False, n_threads=1)
df_urls["prix_min"] = get_min_price(df_urls)

# Get specs for all laptops
df = scrapping.get_all_laptops_specs(df_urls, overwrite=True, n_threads=1)
scrapping.save_images(df_urls)
