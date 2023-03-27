#%%
# this file scrapes the top 10 news articles on google news about the 1619 project from the date it was released until the current date
import pandas as pd
from gnews import GNews
from newspaper import Article
from GoogleNews import GoogleNews
from datetime import datetime, timedelta
import json
#%%
# make a list with all of the dates that you'll want to get information from

def generate_weekly_dates(start_date):
    end_date = datetime.today().date()
    current_date = start_date.date()
    dates = []

    while current_date <= end_date:
        formatted_date = current_date.strftime('%m/%d/%Y')
        dates.append(formatted_date)
        current_date += timedelta(days=7)

    return dates
start_date = datetime(2019, 8, 14)
dates = generate_weekly_dates(start_date)
#%%

def pairs_dict(dates):
    pairs = {}
    for i in range(len(dates)-1):
        pairs[i] = [dates[i], dates[i+1]]
    return pairs

z = pairs_dict(dates)
#%%
z_small = dict(list(z.items())[:3])
#%%
news_results = []
def search_news(news_dict):
    googlenews = GoogleNews()
    for key, values in news_dict.items():
        date1, date2 = values
        googlenews = GoogleNews(start=f'{date1}',end=f'{date2}')
        googlenews.set_encode('utf-8')
        googlenews.search('1619 Project')
        print(f"Searching for news between {date1} and {date2}")
        out = googlenews.results()
        news_results.extend(out)
search_news(z)
news_results

# %%

for news in news_results:
    news['date'] = str(news['date'])
    news['datetime'] = str(news['datetime'])

with open('data1619_weeklyTop10.json', 'w', encoding='utf-8') as f:
    json.dump(test, f, ensure_ascii=False, indent=4)
    
df = pd.DataFrame(news_results)
df.to_csv('data1619_weeklyTop10.csv', index=False, encoding='utf-8-sig')
# %%
