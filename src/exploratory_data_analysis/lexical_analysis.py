"""
Lexical Analysis (Word Level)

This involves digging into the actual language used in the dataset.
● N-gram Analysis: What are the most common pairs (bigrams) or triplets
(trigrams) of words? This reveals common phrases or domain-specific
jargon.

● Stop Word Density: How much of the text is "filler" (the, is, at)? High
density might mean you need more aggressive cleaning.

● Word Clouds & Frequency: A quick visual check to see if the most
frequent words actually align with the task.

"""

import pandas as pd
from nltk.probability import FreqDist
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

class LexicalAnalysis:
  def __init__(self):
    self.punctuations = ['.', ',', '<', '>', '`', '', '``', "'", "'"]
    self.top_words = []
  
  def get_words(self, text):
    if pd.isna(text):
      return []
    words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", str(text).lower())
    return [w for w in words if len(w) > 0 and w not in self.punctuations]

  def get_ngrams(self, tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

  def ngram_analysis(self, df: pd.DataFrame) -> dict:
    """
    Perform n-gram analysis on a DataFrame.
    """
    bigrams = []
    trigrams = []
    for sentence in df['text']:
      if pd.isna(sentence) or not isinstance(sentence, str):
        continue
      words = self.get_words(sentence)

      bigrams.extend(self.get_ngrams(words, 2))
      trigrams.extend(self.get_ngrams(words, 3))
    
    bigram_freq = FreqDist(bigrams)
    trigram_freq = FreqDist(trigrams)

    top_20_bigrams = bigram_freq.most_common(20)
    top_20_trigrams = trigram_freq.most_common(20)

    return {
      'bigram_freq': bigram_freq,
      'trigram_freq': trigram_freq,
      'top_20_bigrams': top_20_bigrams,
      'top_20_trigrams': top_20_trigrams,
    }

  def stop_word_density(self,df: pd.DataFrame, top_20_words: list) -> dict:
    """
    Perform stop word density analysis on a text.
    """
    stop_words = ["the", "is", "at"]
    stop_words.extend(top_20_words)

    stop_word_count = 0
    for sentence in df['text']:
      if pd.isna(sentence) or not isinstance(sentence, str):
        continue
      words = self.get_words(sentence)
      for word in words:
        if word in stop_words:
          stop_word_count += 1
    total_words = sum(len(self.get_words(sentence)) for sentence in df['text'] if pd.notna(sentence) and isinstance(sentence, str))
    return stop_word_count / total_words

  def word_clouds_and_frequency(self, df: pd.DataFrame) -> dict:
    """
    A quick visual check to see if the most frequent words actually align with the task.
    """

    # Word frequency analysis
    words = []
    for sentence in df['text']:
      if pd.isna(sentence) or not isinstance(sentence, str):
        continue
      words.extend(self.get_words(sentence))
    
    word_freq = FreqDist(words)
    top_20_words = word_freq.most_common(20)

    text = ' '.join(words)
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                        max_words=100, colormap='viridis').generate(text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Frequent Words in Corpus')
    plt.tight_layout()
    plt.savefig('wordcloud.png', dpi=150)
    plt.show()

    return {
      'top_20_words': top_20_words,
    }

  def save_lexical_results_to_csv(self, word_clouds_and_frequency, ngram_analysis, stop_word_density, out_dir=None):
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))

    # 1) Top 20 words
    words_df = pd.DataFrame(word_clouds_and_frequency['top_20_words'], columns=['word', 'count'])
    words_df.to_csv(os.path.join(out_dir, 'lexical_top20_words.csv'), index=False)

    # 2) Top 20 bigrams (tuple -> string for CSV)
    bigrams_data = [(f"{a} {b}", c) for (a, b), c in ngram_analysis['top_20_bigrams']]
    bigrams_df = pd.DataFrame(bigrams_data, columns=['bigram', 'count'])
    bigrams_df.to_csv(os.path.join(out_dir, 'lexical_top20_bigrams.csv'), index=False)

    # 3) Top 20 trigrams
    trigrams_data = [(f"{a} {b} {c}", d) for (a, b, c), d in ngram_analysis['top_20_trigrams']]
    trigrams_df = pd.DataFrame(trigrams_data, columns=['trigram', 'count'])
    trigrams_df.to_csv(os.path.join(out_dir, 'lexical_top20_trigrams.csv'), index=False)

    # 4) Stop word density (one-row summary)
    density_df = pd.DataFrame([{'metric': 'stop_word_density', 'value': stop_word_density}])
    density_df.to_csv(os.path.join(out_dir, 'lexical_stop_word_density.csv'), index=False)

  

if __name__ == '__main__':
  df = pd.read_csv('../../data/dontpatronizeme_pcl.tsv', sep='\t', skiprows=4, names=['id', 'art_id', 'keyword', 'country', 'text', 'label'])
  lexical_analysis = LexicalAnalysis()
  ngram_analysis = lexical_analysis.ngram_analysis(df)
  print(f"ngram_analysis: {ngram_analysis}")
  word_clouds_and_frequency = lexical_analysis.word_clouds_and_frequency(df)
  print(f"word_clouds_and_frequency: {word_clouds_and_frequency}")
  top_20_words = [word for word, _ in word_clouds_and_frequency['top_20_words']]
  stop_word_density = lexical_analysis.stop_word_density(df, top_20_words)
  print(f"stop_word_density (the, is, at, {top_20_words}): {stop_word_density}")

  lexical_analysis.save_lexical_results_to_csv(word_clouds_and_frequency, ngram_analysis, stop_word_density)

  print("--------------------------------")
  print("Train and test datasets:")
  train_df = pd.read_csv('../../data/train_semeval_parids-labels.csv')
  dev_df = pd.read_csv('../../data/dev_semeval_parids-labels.csv')
  train_parids = train_df['parids'].tolist()
  dev_parids = dev_df['parids'].tolist()
  