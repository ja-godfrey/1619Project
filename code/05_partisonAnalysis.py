#%%
import pandas as pd 
import ast
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# %%
#open clean, cut
df = pd.read_excel('./../data/derived/03_articles.xlsx')
df.drop(df.columns[[0, 1]], axis=1, inplace=True)
counts = df['media'].value_counts()
df = df[df['media'].isin(counts[counts > 5].index)]

# %%
# create, apply media bias map
outlets = df['media'].unique().tolist()

media_bias_map = {
    "CNN": "D",
    "Education Week": "D",
    "Rolling Stone": "D",
    "The New York Times": "D",
    "PBS": "D",
    "Columbia Journalism Review": "D",
    "USA Today": "D",
    "CBS News": "D",
    "The Wall Street Journal": "R",  # Note: WSJ news is considered more neutral, its editorial/opinion section leans right.
    "WSWS": "D",
    "History News Network": "D",
    "National Review": "R",
    "Forbes": "R",  # Mixed, leans towards conservative in its opinion pieces.
    "NBC News": "D",
    "The Washington Post": "D",
    "WBUR": "D",
    "Milwaukee Independent": "D",
    "NewsOne": "D",
    "The Heritage Foundation": "R",
    "The Texas Tribune": "D",  # Generally considered more neutral but has a slight lean.
    "Pulitzer Center": "D",
    "The Nation": "D",
    "Fox News": "R",
    "Concord Monitor": "D",
    "AP News": "D",  # AP strives for neutrality, but this classification reflects a common perception.
    "Education Next": "R",
    "New York Post": "R",
    "Spokesman Recorder": "D",
    "Poynter": "D",
    "Deseret News": "R",
    "The Daily Signal": "R",
    "Reason Magazine": "R",  # Libertarian, but often classified with conservative media for simplicity.
    "The Federalist": "R",
    "Washington Examiner": "R",
    "Politico": "D",  # Tries for neutral but often perceived as leaning left.
    "The Chronicle of Higher Education": "D",
    "Deadline": "D",
    "Washington Times": "R",
    "NPR": "D",
    "Yahoo News": "D",
    "The Hollywood Reporter": "D",
    "The Hill": "D",  # Tries for neutrality, mixed perceptions.
    "Newsweek": "D",
    "Daily Mail": "R",  # UK-based, but its US coverage often leans right.
    "Axios": "D",  # Strives for neutrality but often seen as leaning left.
    "Los Angeles Times": "D",
    "MSNBC": "D",
    "The Des Moines Register": "D",
    "The Root": "D",
    "Dallas Morning News": "R",  # Endorsement history leans right, though reporting strives for neutrality.
    "The New Yorker": "D",
    "The Daily Beast": "D",
    "Ohio Capital Journal": "D",
    "TIME": "D"
}

df['bias'] = df['media'].map(media_bias_map)

# %%
# get keywords list by bias

df['a_keywords'] = df['a_keywords'].fillna('[]')
df['a_keywords'] = df['a_keywords'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
R_words = sum(df[df['bias'] == 'R']['a_keywords'].tolist(), [])
D_words = sum(df[df['bias'] == 'D']['a_keywords'].tolist(), [])

def calculate_word_percentages(word_list):
    word_counts = Counter(word_list)
    total_words = sum(word_counts.values())
    most_common_words_percentages = [(word, count / total_words * 100) for word, count in word_counts.most_common()]
    
    return most_common_words_percentages

R_output = calculate_word_percentages(R_words)
D_output = calculate_word_percentages(D_words)

# %%
# chart of top 20 keywords by publication bias
R_top20 = R_output[:20]
D_top20 = D_output[:20]

R_words, R_percentages = zip(*R_top20)
D_words, D_percentages = zip(*D_top20)


sns.set_style("white")
colors = {'R': '#E81B23', 'D': '#00AEF3'}

# Adjust positions and width for the bars to make them fatter
positions = np.arange(len(R_top20))
bar_width = 0.35

# Adjust figure size for readability and aesthetics
fig, ax = plt.subplots(figsize=(14, 10))

# Create horizontal bars with added transparency
r_bars = ax.barh(positions - bar_width/2, R_percentages, bar_width, label='R', color=colors['R'], alpha=0.7)
d_bars = ax.barh(positions + bar_width/2, D_percentages, bar_width, label='D', color=colors['D'], alpha=0.7)

# Modern look adjustments
ax.set_xlabel('Percentage', fontsize=12, fontfamily='sans-serif')
ax.set_title('Top 20 keywords by publisher bias', fontsize=20, fontweight='normal', fontfamily='sans-serif')
ax.set_yticks([])

# Custom function to add word labels without y-axis clutter
def add_word_labels(bars, words, offset=(5, 0)):
    for bar, word in zip(bars, words):
        width = bar.get_width()
        ax.annotate(word,
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=offset,
                    textcoords="offset points",
                    ha='left', va='center', fontsize=16, fontfamily='sans-serif')

add_word_labels(r_bars, R_words)
add_word_labels(d_bars, D_words)

# Remove all spines and grid lines for a cleaner look
ax.invert_yaxis()
ax.legend(frameon=False, fontsize=12)
sns.despine(left=False, bottom=False, top=True, right=True)
ax.xaxis.grid(False)
ax.yaxis.grid(False)

plt.tight_layout()
plt.show()

# %%


