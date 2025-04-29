import numpy as np
import torch
import re
import os
import pandas as pd
from collections import OrderedDict
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import pos_tag, word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from symspellpy import SymSpell, Verbosity
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import  KFold
import matplotlib.pyplot as plt
from matplotlib import cm
import spacy
from spacy.matcher import PhraseMatcher
import pytorch_warmup as warmup
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('stopwords')

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
def transform(sentence, word2vec, word2tfidf, dim=50):
    all_doc_tensors = []
    sentence = pd.DataFrame(sentence)
    line_series = sentence['text']
    for each_line in line_series:
        each_line_words = str(each_line).split()
        each_word_embedded_vector = []
        each_word_weight = []
        for word in each_line_words:
            if word in word2vec:
                weight = word2tfidf.get(word, 1.0)
                each_word_embedded_vector.append(word2vec[word] * weight)
                each_word_weight.append(weight)
            elif "_" in word or "-" in word:
                parts = word.split("_") if "_" in word else word.split("-")
                part_vec = [word2vec[p] for p in parts if p in word2vec]
                weight = np.mean([word2tfidf.get(p, 1.0) for p in parts])
                if part_vec:
                    avg_vec = np.mean(part_vec, axis=0)
                    each_word_embedded_vector.append(avg_vec* weight)
                    each_word_weight.append(weight)
                else:
                    # not in the word2vec => use random
                    each_word_embedded_vector.append(np.zeros(dim))
                    each_word_weight.append(weight)
            else:
                # use a random vector for an unknown word
                weight = word2tfidf.get(word, 1.0)
                each_word_embedded_vector.append(np.zeros(dim))
                each_word_weight.append(weight)

        # print("Each word's embedded vector = ", each_word_embedded_vector)
        if len(each_word_embedded_vector) == 0:
            each_sentence_tensor = torch.zeros(dim)
        else:
            tmp_word_arr = np.array(each_word_embedded_vector)
            # each_sentence_tensor = torch.tensor(tmp_word_arr).mean(dim=0)
            each_sentence_tensor = torch.tensor(tmp_word_arr).sum(dim=0)/sum(each_word_weight) # this is the semantic meaning of a sentence
            # each_sentence_tensor = torch.tensor(tmp_word_arr).max(dim=0).values
        all_doc_tensors.append(each_sentence_tensor)
    return torch.stack(all_doc_tensors)

# load csv file
class SentimentDataSet:
    def __init__(self, x_train_file_path, y_train_file_path,x_test_file_path,
                text_column='text', label_column='is_positive_sentiment', transform=None):
        self.word_doc_freq = None
        self.x_train_file_path = x_train_file_path
        self.y_train_file_path = y_train_file_path
        self.x_test_file_path = x_test_file_path
        self.x_col = text_column
        self.y_col = label_column
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.tweet_tokenizer = TweetTokenizer()
        # custom stop words
        new_stopwords = ["oh", "yeah", "so", "just", "and"]
        self.stop_words = set(stopwords.words('english')).union(new_stopwords)
        self.transform = transform
        self.negation_words ={
            "isn't", "wasn't", "aren't", "weren't", "don't", "doesn't", "didn't",
            "can't", "couldn't", "won't", "wouldn't", "shouldn't", "mustn't",
            "mightn't", "shan't", "n't", 'not', 'no', 'never', 'none'
        }
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = PhraseMatcher(self.nlp.vocab)
        self.phrases = [
            # Negation
            "not bad",
            "not good",
            "not impressed",
            "not recommend",
            "no problem",
            "no issues",
            "never again",
            "never disappointed",
            "no complaints",
            "not worth",
            "not happy",
            "not satisfied",
            "no doubt",

            # Positive
            "top notch",
            "well done",
            "highly recommend",
            "five stars",
            "exceeded expectations",
            "good value",
            "user friendly",
            "easy to use",
            "works perfectly",
            "worth every penny",
            "value for money",
            "love it",

            # Negative
            "down the drain",
            "waste of money",
            "fell apart",
            "poor quality",
            "poor product",
            "bad quality",
            "stopped working",
            "bad experience",
            "customer service",
            "never buy",
            "never work",
            "not worth it",
            "cheaply made",
            "broke after",
            "returned it",
            "would not recommend",
            "does not work",
            "out of stock",

            # Neutral
            "fast shipping",
            "as described",
            "packaging was good",
            "looks good",
            "arrived quickly",
        ]
        patterns = [self.nlp.make_doc(phrase) for phrase in self.phrases]
        self.matcher.add("IMPORTANT_PHRASES", patterns)
        self.tfidf_vectorizer = None
        self.word2tfidf = None

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

    def build_document_frequency(self, sentences):
        doc_freq = Counter()
        for sentence in sentences:
            unique_tokens = set(sentence.split())  # only count 1 per doc
            doc_freq.update(unique_tokens)
        # print(doc_freq)
        return doc_freq

    def load_train_csv(self):
        print("Loading train csv file...")
        x_doc = pd.read_csv(self.x_train_file_path)
        y_label = pd.read_csv(self.y_train_file_path)
        x_doc = x_doc[self.x_col]
        y_label = y_label[self.y_col]
        print("Cleaning text...")
        corrected_tokens = x_doc.apply(self.clean_text)
        # turn the token list into a string each row in corrected_tokens
        corrected_str = corrected_tokens.apply(lambda x: " ".join(x))
        self.word_doc_freq = self.build_document_frequency(corrected_str)
        x_doc_clean = corrected_tokens.apply(self.clean_text_with_word_doc_freq) # Series, shape:(2400,)
        print("Finished cleaning text")
        # print(f"x_doc_clean type: {type(x_doc_clean)}, shape: {x_doc_clean.shape}")
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_vectorizer.fit(x_doc_clean.tolist())
        idf_values = dict(zip(self.tfidf_vectorizer.get_feature_names_out(), self.tfidf_vectorizer.idf_))
        self.word2tfidf = idf_values

        X_train = self.transform(x_doc_clean, self.load_glove_as_dict(), self.word2tfidf)
        y_train = torch.tensor(y_label.values, dtype=torch.int64)
        # print("X_train shape = ", X_train.shape)
        # print("X_train = ", X_train)
        # print("y_train shape = ", y_train.shape)
        # print("y_train = ", y_train)
        return X_train, y_train, x_doc_clean

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

    def contraction_filter(self, tokens):
        filtered_tokens = []
        for token in tokens:
            if token in self.negation_words:
                filtered_tokens.append(token)
            else:
                filtered_tokens.append(token)
        return filtered_tokens

    # e.g., "down the drain" => "down_the_drain"
    def preserve_phrases(self, text):
        doc = self.nlp(text)
        matches = self.matcher(doc)
        preserved_text = text
        for match_id, start, end in matches:
            span = doc[start:end]
            # Replace with "down_the_drain"
            preserved_text = preserved_text.replace(span.text, "_".join(span.text.split()))
        return preserved_text

    def clean_text(self, text):
        text = text.strip()
        text = text.lower()
        text = re.sub(r"@\w+", '', text)  # remove @
        text = re.sub(r"#\w+", '', text)  # remove #
        text = re.sub(r'\d+', '', text)  # remove numbers
        text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)  # remove special characters
        text = re.sub(r'\s+', " ", text).strip()  # remove extra spaces
        text = self.preserve_phrases(text)
        tokens = self.tweet_tokenizer.tokenize(text)
        # tokens = word_tokenize(text)
        # tokens = self.contraction_filter(tokens)
        # correct mispronounciation
        corrected_tokens = [self.correct_mispronounciation(token) for token in tokens] # this will change "don't" to "dont"
        return corrected_tokens

    def clean_text_with_word_doc_freq(self, corrected_tokens):
        # remove stop words
        no_stop_words_tokens = [token for token in corrected_tokens
                                if token not in self.stop_words or token in self.negation_words
                                and (self.word_doc_freq.get(token,0) >= 15)]
        # word stemming
        # after_stem = [self.stemmer.stem(each_word) for each_word in corrected_tokens]
        # POS tagging
        tagged_tokens = pos_tag(no_stop_words_tokens)
        # word Lemmatization
        after_lemma = [
                    self.lemmatizer.lemmatize(each_word, self.tag_wordnet_pos(each_tag))
                    for each_word, each_tag in tagged_tokens
        ]

        cleaner_text = " ".join(after_lemma)
        return cleaner_text

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

class CustomNeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CustomNeuralNetwork,self).__init__()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        self.gelu = nn.GELU()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 24)
        self.bn4 = nn.BatchNorm1d(24)
        self.fc5 = nn.Linear(24, 1)
        # self.bn5 = nn.BatchNorm1d(16)
        # self.fc6 = nn.Linear(16, 1)
        # self._init_weights()

    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_normal_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.gelu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.gelu(self.bn2(self.fc2(x)))
        x = self.dropout1(x)
        x = self.gelu(self.bn3(self.fc3(x)))
        x = self.dropout1(x)
        x = self.gelu(self.bn4(self.fc4(x)))
        x = self.dropout1(x)
        # x = self.gelu(self.bn5(self.fc5(x)))
        # x = self.dropout1(x)
        y_pred = torch.sigmoid(self.fc5(x))
        return y_pred

class CustomCNN(nn.Module):
    def __init__(self, input_dim):
        super(CustomCNN, self).__init__()
        self.dropout1 = nn.Dropout(0.7)
        self.dropout2 = nn.Dropout(0.5)
        self.gelu = nn.GELU()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=5, padding=2)  # input_channels=1, output_channels=64
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc = nn.Linear(32 * input_dim, 1)  # Fully connected for final prediction

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = x.unsqueeze(1)  # make it (batch_size, channels=1, input_dim)
        x = self.gelu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.gelu(self.bn2(self.conv2(x)))
        x = self.dropout1(x)
        x = self.gelu(self.bn3(self.conv3(x)))
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)  # flatten for fully connected layer
        y_pred = torch.sigmoid(self.fc(x))
        return y_pred



def train_and_eval(X_train, y_train, x_doc_clean, epochs=30, eta=0.001, batch_size=128, k_folds=5, threshold=0.5):
    print("Start training...")
    kf = KFold(n_splits=k_folds, shuffle=True,random_state=666)
    each_fold_acc = []

    for i, (train_index, val_index) in enumerate(kf.split(X_train)):
        mps_device = torch.device("mps")
        print(f"===================== Fold {i+1} =====================")

        # get the real data from index
        X_train_fold, y_train_fold = X_train[train_index].to(torch.float32), y_train[train_index].to(torch.float32)
        X_val_fold, y_val_fold = X_train[val_index].to(torch.float32), y_train[val_index].to(torch.float32)

        # model init
        # lg_model = LogisticRegressionModel(X_train_fold.shape[1]).to(mps_device).float()
        lg_model = CustomNeuralNetwork(X_train_fold.shape[1]).to(mps_device).float()
        # lg_model = CustomCNN(X_train_fold.shape[1]).to(mps_device).float()
        bce_loss = nn.BCELoss()
        optimizer = optim.AdamW(lg_model.parameters(), lr=eta, weight_decay=1e-5)

        # scheduler_cosine = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        # Warm-up scheduler
        # warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

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
                loss.backward()
                print(f"Fold: {i+1} Epoch:{epoch+1} Batch: {j//batch_size+1} - Loss: {loss.item():.5f}")
                torch.nn.utils.clip_grad_norm_(lg_model.parameters(), 1.0)
                optimizer.step()
                # if warmup_scheduler.dampen():
                #     pass
                # else:
                #     scheduler_cosine.step()

        # validation
        lg_model.eval()
        with torch.no_grad():
            y_val_pred_raw = lg_model(X_val_fold.to(mps_device)).squeeze()
            y_val_pred = (y_val_pred_raw > threshold).float()
            accuracy = (y_val_pred.cpu() == y_val_fold).float().mean().item()
            print(f"Fold {i+1} - Validation accuracy: {accuracy:.5f}")
            each_fold_acc.append(accuracy)

    # avg accuracy of 5 folds
    avg_acc_5_fold = np.mean(each_fold_acc)
    print(f"Average accuracy of 5 folds: {avg_acc_5_fold:.5f}")
    with open("base_line_acc.txt", "a") as f:
        # Append function parameters to the accuracy file
        isCustom = True
        f.write(f"epoch: {epochs}, eta: {eta}, batch_size: {batch_size}, k_folds: {k_folds}, threshold: {threshold}, Custom: {isCustom}\n")
        f.write(f"Average accuracy of 5 folds: {avg_acc_5_fold:.5f}\n")
    return each_fold_acc

if __name__ == '__main__':
    x_train_path = "../train_data/x_train.csv"
    y_train_path = "../train_data/y_train.csv"
    x_test_path = "../test_data/x_test.csv"

    transform_glove_to_dictionary()
    X_train, y_train, x_doc_clean = SentimentDataSet(x_train_path, y_train_path, x_test_path, transform=transform).load_train_csv()
    X_train = X_train.to(torch.float32)
    y_train = y_train.to(torch.float32)
    fold5_acc = train_and_eval(X_train, y_train, x_doc_clean)
