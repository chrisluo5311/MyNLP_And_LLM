import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample documents
documents = [
    "the movie was awesome and inspiring",
    "the movie was awful and boring",
    "awesome acting and good story",
]

stop_words_list = ["the", "was", "and", "a", "an"]
# Initialize the vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents into TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)
tfidf_np = numpy.array(tfidf_matrix.toarray())

# Convert to DataFrame for better readability
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
# tfidf_np = tfidf_df.to_numpy()

# Display the resulting TF-IDF matrix
# print("TF-IDF matrix as DataFrame:")
# print(tfidf_df.shape)
# print(tfidf_df)
print("TF-IDF matrix as numpy array:")
print(tfidf_np.shape)
print(tfidf_np)
tfidf_df.to_csv("sample_tfidf_matrix.csv", index=True)