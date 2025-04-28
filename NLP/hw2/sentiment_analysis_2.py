import numpy as np
import torch
import re
import os
import pandas as pd
from collections import OrderedDict
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from symspellpy import SymSpell, Verbosity
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import  KFold
import nltk

# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger_eng')

# init SymSpell
symSpell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
# "frequency_dictionary_en_82_765.txt" too bad => not using anymore
#term_index=0: Specifies that the first column (index 0) in the file contains the terms (words).
#count_index=1: Specifies that the second column (index 1) in the file contains the frequency counts of the terms.
symSpell.load_dictionary("./glove_symspell_dictionary.txt", term_index=0, count_index=1)

# glove path
glove_file_path = "./glove.6B.50d.txt"
# glove as dictionary path
glove_symspell_file_path = "./glove_symspell_dictionary.txt"

# transform glove to dictionary for misspelling correction
def transform_glove_to_dictionary():
    if not os.path.exists(glove_symspell_file_path) or os.path.getsize(glove_symspell_file_path) == 0:
        with open(glove_file_path, "r") as input_file, open(glove_symspell_file_path, "w") as output_file:
            for each_line in input_file:
                first_word = each_line.split()[0]
                output_file.write(f"{first_word} 1\n")
        print("Successfully transformed glove to dictionary ✅")
    else:
        print("glove_symspell_file_path exists => skip")

# word embeddings dimension 50
def transform(sentence, word2vec, dim=50):
    all_doc_tensors = []
    sentence = pd.DataFrame(sentence)
    line_series = sentence['text']
    for each_line in line_series:
        each_line_words = str(each_line).split()
        each_word_embedded_vector = [word2vec[each_word] for each_word in each_line_words if each_word in word2vec]
        # print("Each word's embedded vector = ", each_word_embedded_vector)
        if len(each_word_embedded_vector) == 0:
            each_sentence_tensor = torch.zeros(dim)
        else:
            tmp_np_arr = np.array([vec for vec in each_word_embedded_vector])
            each_sentence_tensor = torch.tensor(tmp_np_arr).mean(dim=0)
            # print(f"👀Sentence vector:\n{each_sentence_tensor}")
            # each_sentence_tensor = torch.mean(torch.stack(each_word_embedded_vector), dim=0)
            # could use torch.max to capture the strongest signal and concat with torch.mean
            # each_sentence_tensor = torch.max(torch.stack(each_word_embedded_vector), dim=0)
        all_doc_tensors.append(each_sentence_tensor)
    return torch.stack(all_doc_tensors)

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

    def load_glove_as_dict(self):
        print("Loading glove as dictionary...")
        word_embeddings = pd.read_csv('./glove.6B.50d.txt.zip',
                                        header=None, sep=' ', index_col=0,
                                        compression='zip', encoding='utf-8', quoting=3)
        # Build a dict that will map from string word to 50-dim vector
        word_list = word_embeddings.index.values.tolist()
        word2vec = OrderedDict(zip(word_list, word_embeddings.values))
        print("Type of word2vec = ", type(word2vec))
        print("len(word2vec) = ", len(word2vec))
        return word2vec

    def load_train_csv(self):
        print("Loading train csv file...")
        x_doc = pd.read_csv(self.x_train_file_path)
        y_label = pd.read_csv(self.y_train_file_path)
        x_doc = x_doc[self.x_col]
        y_label = y_label[self.y_col]
        print("Cleaning text...")
        x_doc_clean = x_doc.apply(self.clean_text)
        X_train = self.transform(x_doc_clean, self.load_glove_as_dict())
        y_train = torch.tensor(y_label.values, dtype=torch.int64)
        print("X_train shape = ", X_train.shape)
        print("X_train = ", X_train)
        print("y_train shape = ", y_train.shape)
        print("y_train = ", y_train)
        return X_train, y_train

    def correct_mispronounciation(self, word):
        # Verbosity.CLOSEST: A parameter specifying that the method should return the closest match
        # max_edit_distance: Limits the maximum number of character edits (insertions, deletions, substitutions, or transpositions) allowed to consider a match.
        possible_answer = symSpell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        if possible_answer:
            return possible_answer[0].term
        return word

    def tag_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def clean_text(self, text):
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
        # POS tagging
        tagged_tokens = pos_tag(corrected_tokens)
        # word Lemmatization
        after_lemma = [self.lemmatizer.lemmatize(each_word, self.tag_wordnet_pos(each_tag)) for each_word, each_tag in tagged_tokens]
        cleaner_text = " ".join(after_lemma)
        return cleaner_text

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

def CustomNeuralNetwork():
    def __init__(self, input_dim):
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        y_pred = torch.sigmoid(self.fc4(x))
        return y_pred

def train_and_eval(X_train, y_train, epochs=30, eta=0.001, batch_size=16, k_folds=5, threshold=0.5):
    print("Start training...")
    kf = KFold(n_splits=k_folds, shuffle=True,random_state=666)
    each_fold_acc = []

    for i, (train_index, val_index) in enumerate(kf.split(X_train)):
        mps_device = torch.device("mps")
        print(f"===================== Fold {i} =====================")

        # get the real data from index
        X_train_fold, y_train_fold = X_train[train_index].to(torch.float32), y_train[train_index].to(torch.float32)
        X_val_fold, y_val_fold = X_train[val_index].to(torch.float32), y_train[val_index].to(torch.float32)

        # model init
        lg_model = LogisticRegressionModel(X_train_fold.shape[1]).to(mps_device).float()
        bce_loss = nn.BCELoss()
        optimizer = optim.Adam(lg_model.parameters(), lr=eta)

        for epoch in range(epochs):
            print(f"################## Epoch: {epoch+1} ##################")
            lg_model.train()
            shuffle_index = torch.randperm(X_train_fold.shape[0])
            for j in range(0, X_train_fold.shape[0], batch_size):
                batch_indexes = shuffle_index[j: j+batch_size]
                X_batch, y_batch = X_train_fold[batch_indexes].to(mps_device), y_train_fold[batch_indexes].unsqueeze(1).to(mps_device)
                optimizer.zero_grad()
                y_pred = lg_model(X_batch)
                loss = bce_loss(y_pred, y_batch)
                # current batch's loss
                print(f"Batch {j//batch_size+1} - Loss: {loss.item():.5f}")
                loss.backward()
                optimizer.step()

        # validation
        lg_model.eval()
        with torch.no_grad():
            y_val_pred_raw = lg_model(X_val_fold.to(mps_device)).squeeze()
            y_val_pred = (y_val_pred_raw > threshold).float()
            accuracy = (y_val_pred.cpu() == y_val_fold).float().mean().item()
            print(f"Fold {i} - Validation accuracy: {accuracy:.5f}")
            each_fold_acc.append(accuracy)

    # avg accuracy of 5 folds
    avg_acc_5_fold = np.mean(each_fold_acc)
    print(f"Average accuracy of 5 folds: {avg_acc_5_fold:.5f}")
    with open("base_line_acc.txt", "a") as f:
        # Append function parameters to the accuracy file
        f.write(f"epoch: {epochs}, eta: {eta}, batch_size: {batch_size}, k_folds: {k_folds}, threshold: {threshold}\n")
        f.write(f"Average accuracy of 5 folds: {avg_acc_5_fold:.5f}\n")
    return each_fold_acc

if __name__ == '__main__':
    x_train_path = "../train_data/x_train.csv"
    y_train_path = "../train_data/y_train.csv"
    x_test_path = "../test_data/x_test.csv"

    transform_glove_to_dictionary()
    X_train, y_train = SentimentDataSet(x_train_path, y_train_path, x_test_path, transform=transform).load_train_csv()
    X_train = X_train.to(torch.float32)
    y_train = y_train.to(torch.float32)
    fold5_acc = train_and_eval(X_train, y_train)
