import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import nltk
import re
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('universal_tagset')


"""
3. Semantic & Syntactic Exploration
Modern NLP requires understanding the "meaning" behind the statistics.
● Part-of-Speech (POS) Tagging: Are there more verbs than nouns? (e.g., in
instruction-following tasks, verbs are dominant).
● Named Entity Recognition (NER): Does the dataset focus on specific
people, locations, or organizations?
● Embedding Visualization: Using techniques like t-SNE or UMAP to
project high-dimensional word vectors into 2D space. This allows you to
see if similar concepts are naturally clustering together before you even
train a model.
"""

class SemanticSyntacticExploration:
  def __init__(self):
    self.punctuations = ['.', ',', '<', '>', '`', '', '``', "'", "'"]

  def get_words(self, text):
    if pd.isna(text):
      return []
    words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", str(text).lower())
    return [w for w in words if len(w) > 0]

  def part_of_speech_tagging(self, df: pd.DataFrame) -> dict:
    """
    Part-of-Speech (POS) Tagging: Are there more verbs than nouns? (e.g., in
    instruction-following tasks, verbs are dominant).
    """

    pos_tags = []
    verb_count = 0
    noun_count = 0

    for sentence in df:
        if pd.isna(sentence) or not isinstance(sentence, str):
            continue
        words = self.get_words(sentence)
        sent_tags = pos_tag(words, tagset='universal')  # Use universal tagset
        pos_tags.extend(sent_tags)
        for word, pos in sent_tags:  # Count per sentence (or use pos_tags for total)
            if pos == 'VERB':
                verb_count += 1
            elif pos == 'NOUN':
                noun_count += 1

    return {
        'verb_count': verb_count,
        'noun_count': noun_count,
    }

  def named_entity_recognition(self, df: pd.DataFrame) -> dict:
    """
    Named Entity Recognition (NER): Does the dataset focus on specific
    people, locations, or organizations?
    """
    pass

if __name__ == '__main__':
  df = pd.read_csv('../../data/dontpatronizeme_pcl.tsv', sep='\t', skiprows=4, names=['id', 'art_id', 'keyword', 'country', 'text', 'label'])
  semantic_syntactic_exploration = SemanticSyntacticExploration()
  part_of_speech_tagging = semantic_syntactic_exploration.part_of_speech_tagging(df['text'])
  print(part_of_speech_tagging)
  named_entity_recognition = semantic_syntactic_exploration.named_entity_recognition(df['text'])
  print(named_entity_recognition)