
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ast

def common_keywords_graph(number_of_keywords):
    # clean
    df = pd.read_excel('./../data/derived/articles.xlsx')
    df.dropna(subset=['a_keywords'], inplace=True)
    df['a_keywords'] = df['a_keywords'].apply(lambda x: ast.literal_eval(x)) 
    
    # most common words in keywords
    all_keywords = [keyword for keywords_list in df['a_keywords'] for keyword in keywords_list]

    # count the frequency of each keyword
    keyword_counts = Counter(all_keywords)

    # get the 10 most common keywords
    top_keywords = keyword_counts.most_common(number_of_keywords)

    # create a bar chart of the top keywords
    fig, ax = plt.subplots()
    ax.bar([x[0] for x in top_keywords], [x[1] for x in top_keywords])
    ax.set_title('Most common keywords in articles about 1619 Project')
    ax.set_xlabel('Keyword')
    ax.set_ylabel('Count')
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.show()


if __name__ == "__main__":
    common_keywords_graph(20)