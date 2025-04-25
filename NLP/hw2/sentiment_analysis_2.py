import torch
import re
import os
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from symspellpy import SymSpell, Verbosity
import nltk

# nltk.download('punkt_tab')
# init SymSpell
symSpell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
# too bad => not using anymore
# symSpell.load_dictionary("./frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)

# glove path
glove_file_path = "./glove.6B.50d.txt"
# glove as dictionary path
glove_symspell_file_path = "./glove_symspell_dictionary.txt"

# transform glove to dictionary for misspelling correction
def transform_glove_to_dictionary():
    if not os.path.exists(glove_file_path):
        with open(glove_file_path, "r") as input_file, open(glove_symspell_file_path, "w") as output_file:
            for each_line in input_file:
                first_word = each_line.strip()[0]
                output_file.write(f"{first_word} 1\n")
        print("Successfully transformed glove to dictionary ✅")
    else:
        print("glove_symspell_file_path exists => skip")

    print("Finished loading glove_symspell_file_path ✅")

# load csv file
class SentimentDataSet:
    def __init__(self, x_train_file_path, y_train_file_path,x_test_file_path,
                text_column='text', label_column='is_positive_sentiment', transform=None):
        self.x_train_file_path = x_train_file_path
        self.y_train_file_path = y_train_file_path
        self.x_test_file_path = x_test_file_path
        self.x_col = text_column
        self.y_col = label_column
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.transform = transform

    def load_train_csv(self):
        print("Loading train csv file...")
        x_doc = pd.read_csv(self.x_train_file_path)
        y_label = pd.read_csv(self.y_train_file_path)
        x_doc = x_doc[self.x_col]
        y_label = y_label[self.y_col]
        x_doc_clean = x_doc.apply(self.clean_text)
        print(x_doc_clean)

    def correct_mispronounciation(self, word):
        possible_answer = symSpell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        if possible_answer:
            return possible_answer[0].term
        return word

    def clean_text(self, text):
        print("Cleaning text...")
        text = text.strip()
        text = text.lower()
        text = re.sub(r"@\w+", '', text)  # remove @
        text = re.sub(r"#\w+", '', text)  # remove #
        text = re.sub(r'\d+', '', text)  # remove numbers
        text = re.sub(r'[^a-zA-Z0-9\s]', " ", text)  # remove special characters
        text = re.sub(r'\s+', " ", text).strip()  # remove extra spaces
        tokens = word_tokenize(text)
        # correct mispronounciation
        corrected_tokens = [self.correct_mispronounciation(token) for token in tokens]
        # word stemming
        # after_stem = [self.stemmer.stem(each_word) for each_word in corrected_tokens]
        # word Lemmatization
        after_lemma = [self.lemmatizer.lemmatize(each_word) for each_word in corrected_tokens]
        cleaner_text = " ".join(after_lemma)
        return cleaner_text




if __name__ == '__main__':
    x_train_path = "../train_data/x_train.csv"
    y_train_path = "../train_data/y_train.csv"
    x_test_path = "../test_data/x_test.csv"

    transform_glove_to_dictionary()
    # SentimentDataSet(x_train_path, y_train_path, x_test_path).load_train_csv()