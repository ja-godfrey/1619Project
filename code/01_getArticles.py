#%%
import pandas as pd
from newspaper import Article
#%%

df = pd.read_csv('../data/raw/data1619_weeklytop10.csv')
# %%
url = df['link'][0]
article = Article(url)
# %%
article.download()

# %%
article.parse()
# %%
article.authors
# %%
article.text
# %%
# https://newspaper.readthedocs.io/en/latest/

for i in range(len(df['link'])):
    url = df['link'][i]
    print(url)
# %%
