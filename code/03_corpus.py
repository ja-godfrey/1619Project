#%%
import pandas as pd
import re
import os

df = pd.read_excel('./../data/derived/articles.xlsx')

#%%
# how many words are in the corpus?
df.drop_duplicates(subset='a_text', inplace=True)
df.dropna(subset=['a_text'], inplace=True)
df = df[df['a_text'].str.len() >= 200]
total_words = sum(df['a_text'].str.split().str.len())
total_words
#%%
# get all the images
# https://levelup.gitconnected.com/how-to-download-images-from-urls-convert-the-type-and-save-them-in-the-cloud-with-python-294e11811243
for url in df['a_image']:
    try: 
        # get image title
        row = df.loc[df['a_image'] == url]
        title = row['a_title'].iloc[0]
        clean_title = re.sub('[^a-zA-Z]+', '', title).replace(' ', '_')
        clean_title = clean_title[:20]
        image_folder = "./../data/raw/img/"

        response = requests.get(url)
        img_ext = mimetypes.guess_extension(content_type)
        file_name = image_folder + clean_title + img_ext

        with open(file_name, "wb") as f_imag:
            f_imag.write(response.content)
    except Exception as e:
        print(f'Error {e} downloading image {url}')
    

# %%
# save new dataframe

df['a_title'] = df['a_title'].astype(str)

def clean_title(title):
    # Remove non-letter characters and replace spaces with '_'
    title = re.sub(r'[^a-zA-Z\s]', '', title).replace(' ', '')
    # Limit to first 20 characters
    title = title[:20]
    return title

df['clean_title'] = df['a_title'].apply(clean_title)
df.to_excel('./../data/derived/03_articles.xlsx')
# %%
