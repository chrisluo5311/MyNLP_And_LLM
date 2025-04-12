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
vectorizer = TfidfVectorizer(stop_words=stop_words_list)

# Fit and transform the documents into TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)

# Convert to DataFrame for better readability
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Display the resulting TF-IDF matrix
print(tfidf_df)
tfidf_df.to_csv("sample_tfidf_matrix.csv", index=True)