#%%
import pandas as pd
from newspaper import Article
from newspaper import Config

# user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'

config = Config()
config.browser_user_agent = user_agent

df = pd.read_csv('../data/raw/data1619_weeklytop10.csv')
# %%

# create a list of column names
new_cols = ['a_authors', 'a_text', 'a_title', 'a_image', 'a_movies', 'a_keywords']

# use the assign method to add the new columns
df[new_cols] = pd.DataFrame([['']*len(new_cols)], columns=new_cols)

for i in range(len(df['link'])):
    try:
        url = df['link'][i]
        article = Article(url)
        article.download()
        article.parse()
        df['a_authors'][i] = article.authors
        df['a_text'][i] = article.text
        df['a_title'][i] = article.title
        df['a_image'][i] = article.top_image
        df['a_movies'][i] = article.movies
        article.nlp()
        df['a_keywords'][i] = article.keywords

        print(f'article {i}: authors:{article.authors}, title:{article.title}, image:{article.top_image}, movies:{article.movies}, keywords{article.keywords}')
    except:
        pass

# %%
print(f'article {i}: title:{article.authors}, image:{article.top_image}, movies:{article.movies}, keywords{article.keywords}')
# %%
df.to_excel('./../data/derived/articles.xlsx')
# %%
