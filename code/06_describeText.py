#%%
import pandas as pd
# %%

df = pd.read_json('./../data/derived/04_partison.json')
df = df.loc[df['a_text'].str.len().between(500, 20000)]
df.drop_duplicates(subset=['a_text'], inplace=True)
df = df[df['a_text'].apply(lambda x: isinstance(x, str) and x.strip() != '')]
# %%
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string

# Download necessary NLTK data (do this once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Calculate text lengths
df['text_length'] = df['a_text'].apply(len)

# Tokenize and clean words (remove punctuation and stopwords)
stop_words = set(stopwords.words('english'))
df['cleaned_words'] = df['a_text'].apply(lambda x: [word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words])

# Calculate most common words across all rows
all_words = [word for sublist in df['cleaned_words'].tolist() for word in sublist]
word_counts = Counter(all_words)
most_common_words = word_counts.most_common(10)  # Adjust number as needed

# POS tagging
df['pos_tags'] = df['cleaned_words'].apply(nltk.pos_tag)

# %%

import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set_style("whitegrid")

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='text_length', hue='bias', kde=True, element='step', palette={'R': 'red', 'D': 'blue'})
plt.title('Distribution of Text Lengths by Bias')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.savefig('./../figures/textlength.png', dpi=300, bbox_inches='tight')
plt.show()
# %%
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='text_length', hue='bias', kde=True, element='step', 
             palette={'R': 'red', 'D': 'blue'}, stat='density', common_norm=False)
plt.title('Percentage Distribution of Text Lengths by Bias')
plt.xlabel('Text Length')
plt.ylabel('Density')
plt.savefig('./../figures/textdensity.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
import textstat

# Calculate the Flesch Reading Ease score for each text entry
df['reading_level'] = df['a_text'].apply(textstat.flesch_reading_ease)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='reading_level', hue='bias', kde=True, element='step',
             palette={'R': 'red', 'D': 'blue'}, stat='density', common_norm=False)
plt.title('Distribution of Flesch Reading Ease Scores by Bias')
plt.xlabel('Flesch Reading Ease Score')
plt.ylabel('Density')
plt.savefig('./../figures/readingEase.png', dpi=300, bbox_inches='tight')
plt.show()

# %%

# Calculate the Flesch-Kincaid Grade Level for each text entry
df['reading_grade_level'] = df['a_text'].apply(textstat.flesch_kincaid_grade)
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='reading_grade_level', hue='bias', kde=True, element='step',
             palette={'R': 'red', 'D': 'blue'}, stat='density', common_norm=False)
plt.title('Distribution of Flesch-Kincaid Reading Grade Levels by Bias')
plt.xlabel('Flesch-Kincaid Reading Grade Level')
plt.ylabel('Density')
plt.savefig('./../figures/gradeLevel.png', dpi=300, bbox_inches='tight')
plt.show()


# %%
# sentiment analysis
from textblob import TextBlob

# Function to calculate sentiment
def calculate_sentiment(text):
    # Create a TextBlob object
    blob = TextBlob(text)
    # Return sentiment
    return blob.sentiment

# Apply the function to calculate sentiment
df['sentiment'] = df['a_text'].apply(lambda x: calculate_sentiment(x))

# The sentiment column will now have tuples of the form (polarity, subjectivity)
# You can split these into separate columns if desired
df['polarity'] = df['sentiment'].apply(lambda x: x.polarity)
df['subjectivity'] = df['sentiment'].apply(lambda x: x.subjectivity)

import matplotlib.pyplot as plt
import seaborn as sns

# Polarity distribution
plt.figure(figsize=(12, 8))
sns.histplot(data=df, x='polarity', hue='bias', bins=20, kde=True, element='step', palette={'R': 'red', 'D': 'blue'}, stat='density', common_norm=False)
plt.title('Distribution of Polarity Scores by Bias')
plt.xlabel('Polarity')
plt.ylabel('Density')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:], title='Bias')
plt.savefig('./../figures/polarity.png', dpi=300, bbox_inches='tight')
plt.show()

# Subjectivity distribution
plt.figure(figsize=(12, 8))
sns.histplot(data=df, x='subjectivity', hue='bias', bins=20, kde=True, element='step', palette={'R': 'red', 'D': 'blue'}, stat='density', common_norm=False)
plt.title('Distribution of Subjectivity Scores by Bias')
plt.xlabel('Subjectivity')
plt.ylabel('Density')
plt.legend(title='Bias')
plt.savefig('./../figures/subjectivity.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# LIWC
from textblob import TextBlob

# Example of sentiment analysis
df['sentiment'] = df['a_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
# Polarity ranges from -1 (negative) to 1 (positive)

# Adding subjectivity score
df['subjectivity'] = df['a_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
# Subjectivity ranges from 0 (objective) to 1 (subjective)
import spacy

nlp = spacy.load('en_core_web_sm')  # Make sure to download this model first

# Named Entity Recognition (NER)
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

df['named_entities'] = df['a_text'].apply(extract_entities)

# Function to calculate percentages and sort entity types by bias
def entity_type_percentages_by_bias(df, bias_value):
    bias_entities = [entity for sublist in df[df['bias'] == bias_value]['named_entities'].tolist() for entity in sublist]
    entity_types = [entity_type for _, entity_type in bias_entities]
    entity_type_counts = Counter(entity_types)
    total_count = sum(entity_type_counts.values())
    percentages = {entity_type: (count / total_count) * 100 for entity_type, count in entity_type_counts.items()}
    # Sort by percentage in descending order
    sorted_percentages = dict(sorted(percentages.items(), key=lambda item: item[1], reverse=True))
    return sorted_percentages

def add_percentage_labels(ax, df):
    for i, (percentage, entity_type) in enumerate(zip(df['Percentage'], df['Entity Type'])):
        offset = max(percentage * 0.01, 0.5)
        label_position = percentage + offset
        ax.text(label_position, i, f'{percentage:.0f}%', va='center')

# Calculate percentages for each bias
r_entity_percentages = entity_type_percentages_by_bias(df, 'R')
d_entity_percentages = entity_type_percentages_by_bias(df, 'D')

# Convert to DataFrame for easier plotting
r_df = pd.DataFrame(list(r_entity_percentages.items()), columns=['Entity Type', 'Percentage'])
d_df = pd.DataFrame(list(d_entity_percentages.items()), columns=['Entity Type', 'Percentage'])

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
fig.suptitle('Percentage Distribution of Named Entity Types by Bias', size=25)

# Plot for 'R' bias
sns.barplot(x='Percentage', y='Entity Type', data=r_df, ax=axes[0], color='red')
axes[0].set_title('Bias R')
axes[0].set_xlabel('Percentage')
axes[0].set_ylabel('Entity Type')
axes[0].set_xticks([0, 5, 10, 15, 20])
add_percentage_labels(axes[0], r_df)

# Plot for 'D' bias
sns.barplot(x='Percentage', y='Entity Type', data=d_df, ax=axes[1], color='blue')
axes[1].set_title('Bias D')
axes[1].set_xlabel('Percentage')
axes[1].set_ylabel('')
axes[1].set_xticks([0, 5, 10, 15, 20])
add_percentage_labels(axes[1], d_df)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('./../figures/NER.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# topic modeling

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Splitting the dataset based on 'bias'
df_r = df[df['bias'] == 'R']['a_text']
df_d = df[df['bias'] == 'D']['a_text']

# Function to apply LDA topic modeling
def apply_lda(texts, n_topics=5):
    # Vectorize the text data
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(texts)
    
    # Apply LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(dtm)
    
    # Display topics and their top words
    words = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        print(f"Topic #{topic_idx+1}:")
        print(" ".join([words[i] for i in topic.argsort()[:-11:-1]]))
        print("\n")

# Number of topics
n_topics = 5

# Perform LDA for 'R' bias
print("Topics for 'R' Bias:\n")
apply_lda(df_r, n_topics=n_topics)

# Perform LDA for 'D' bias
print("\nTopics for 'D' Bias:\n")
apply_lda(df_d, n_topics=n_topics)
# %%
