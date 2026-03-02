# Basic Statistical Profiling

import pandas as pd
from transformers import AutoTokenizer
import re

import ast
from pathlib import Path

def analyse_semeval_label_distribution(data_dir=None):
    """
    Analyse label distribution in train and dev SemEval label files.
    Labels are 7-dimensional binary vectors.
    non-PCL = all zeros; PCL (positive) = at least one dimension is 1.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent.parent / 'data'
    train = pd.read_csv(data_dir / 'train_semeval_parids-labels.csv')
    dev = pd.read_csv(data_dir / 'dev_semeval_parids-labels.csv')

    def parse_label(s):
        return ast.literal_eval(s)

    for name, df in [('train', train), ('dev', dev)]:
        df['label_vec'] = df['label'].apply(parse_label)
        df['num_positive'] = df['label_vec'].apply(sum)
        # non-PCL = all zeroes; PCL = everything else
        df['is_pcl'] = df['num_positive'] > 0

    n_dims = 7
    results = []
    for split_name, df in [('train', train), ('dev', dev)]:
        n = len(df)
        # Binary: non-PCL vs PCL
        n_non_pcl = (~df['is_pcl']).sum()
        n_pcl = df['is_pcl'].sum()
        results.append({
            'split': split_name,
            'dimension': 'non-PCL',
            'count_positive': int(n_non_pcl),
            'fraction_positive': n_non_pcl / n if n else 0,
        })
        results.append({
            'split': split_name,
            'dimension': 'PCL',
            'count_positive': int(n_pcl),
            'fraction_positive': n_pcl / n if n else 0,
        })
        # Per-dimension: fraction of samples with category i = 1
        for dim in range(n_dims):
            count = df['label_vec'].apply(lambda v: v[dim]).sum()
            results.append({
                'split': split_name,
                'dimension': f'category_{dim}',
                'count_positive': int(count),
                'fraction_positive': count / n if n else 0,
            })
        # Distribution of number of PCL categories per sample (0, 1, ..., 7)
        counts_per_sample = df['num_positive'].value_counts().sort_index()
        for k, v in counts_per_sample.items():
            results.append({
                'split': split_name,
                'dimension': f'num_categories_{k}',
                'count_positive': int(v),
                'fraction_positive': v / n if n else 0,
            })

    results_df = pd.DataFrame(results)
    out_path = Path(__file__).resolve().parent / 'semeval_label_distribution.csv'
    results_df.to_csv(out_path, index=False)
    return {'train': train, 'dev': dev, 'results_df': results_df}

class BasicStatisticalProfiling:
  def __init__(self):
    self.punctuations = ['.', ',', '<', '>', '`', "","" '``', "'", "'"]

  def get_words(self, text):
    if pd.isna(text):
      return []
    words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", str(text).lower())
    return [w for w in words if len(w) > 0 and w not in self.punctuations]

  def basic_statistical_profiling(self, df: pd.DataFrame) -> dict:
    """
    Perform basic statistical profiling on a DataFrame.
    ● Token Count: What is the average, minimum, and maximum sentence length?   

    ● Vocabulary Size: 
      - How many unique words exist? 
      - This dictates the size of your embedding layer.  

    ● Class Distribution: 
      - Is the dataset balanced? 
      - (e.g., In a hate speech task, if 98% of the data is "Non-Toxic," 
        your model might achieve 98% accuracy just by guessing "Non-Toxic" every time). 
    """

    # Token count
    token_count = 0
    min_sentence_length = float('inf')
    max_sentence_length = 0
    total_sentence_length = 0
    vocabulary = set()

    for sentence in df['text']:
      if pd.isna(sentence) or not isinstance(sentence, str):
        continue
      words = self.get_words(sentence)

      sentence_length = len(words)
      min_sentence_length = min(min_sentence_length, sentence_length)
      max_sentence_length = max(max_sentence_length, sentence_length)
      total_sentence_length += len(words)
      token_count += len(words)
      vocabulary.update(words)

    average_sentence_length = total_sentence_length / len(df['text'])

    label_percentages = {}
    for label in df['label'].unique():
      label_count = len(df[df['label'].isin([0,1])])
      label_percentages['non-PCL'] = label_count / len(df['label'])
      label_count = len(df[df['label'].isin([2,3,4])])
      label_percentages['PCL'] = label_count / len(df['label'])

    return {
      'token_count': token_count,
      'vocabulary_size': len(vocabulary),
      'average_sentence_length': average_sentence_length,
      'min_sentence_length': min_sentence_length,
      'max_sentence_length': max_sentence_length,
      'label_percentages': label_percentages,
    }

if __name__ == '__main__':
  bsp = BasicStatisticalProfiling()
  df = pd.read_csv('../../data/dontpatronizeme_pcl.tsv', sep='\t', skiprows=4, names=['id', 'art_id', 'keyword', 'country', 'text', 'label'])
  profile = bsp.basic_statistical_profiling(df)
  # Flatten for CSV (label_percentages becomes separate columns)
  flat = {
    'token_count': profile['token_count'],
    'vocabulary_size': profile['vocabulary_size'],
    'average_sentence_length': profile['average_sentence_length'],
    'min_sentence_length': profile['min_sentence_length'],
    'max_sentence_length': profile['max_sentence_length'],
  }
  for label, pct in profile['label_percentages'].items():
    flat[f'label_{label}_percentage'] = pct
  
  df_out = pd.DataFrame([flat])
  df_out.to_csv('basic_statistical_profiling_results.csv', index=False)
  print(profile)

  # After loading df and computing length per row
  df['word_count'] = df['text'].apply(lambda x: len(bsp.get_words(x)) if pd.notna(x) else 0 and x not in bsp.punctuations)
  short = df[df['word_count'] >= 512]
  print("Samples over the length of 512 words:")
  print(f"Count: {len(short)}")
  print(short[['text', 'label']])

  semeval_dist = analyse_semeval_label_distribution()
  print("SemEval label distribution (per dimension and per num_labels):")
  print(semeval_dist['results_df'])
  print(f"Saved to semeval_label_distribution.csv")