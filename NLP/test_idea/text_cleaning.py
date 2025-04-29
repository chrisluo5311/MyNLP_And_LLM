from nltk.tokenize import TweetTokenizer
from symspellpy import SymSpell, Verbosity
import re
import spacy
from spacy.matcher import PhraseMatcher

text_list = ["THAT one didn't work either", "I'm very disappointed with my decision", "Absolutel junk",
            "It dit not work most of the time with my Nokia 5320.", "down the drain", "out of hand",
            "In short - this was a monumental waste of time and energy and I would not recommend anyone to EVER see this film.",
            "My sashimi was poor quality being soggy and tasteless."]

symSpell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
symSpell.load_dictionary("./glove_symspell_dictionary.txt", term_index=0, count_index=1)
def correct_mispronounciation(word):
    # Verbosity.CLOSEST: A parameter specifying that the method should return the closest match
    # max_edit_distance: Limits the maximum number of character edits (insertions, deletions, substitutions, or transpositions) allowed to consider a match.
    possible_answer = symSpell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
    if possible_answer:
        return possible_answer[0].term
    return word

negation_words = {
    "isn't", "wasn't", "aren't", "weren't", "don't", "doesn't", "didn't",
    "can't", "couldn't", "won't", "wouldn't", "shouldn't", "mustn't",
    "mightn't", "shan't", "n't"
}
def contraction_filter(tokens):
    filtered_tokens = []
    for token in tokens:
        if token in negation_words:
            filtered_tokens.append(token)
        elif "'" in token and token not in negation_words:
            continue
        else:
            filtered_tokens.append(token)
    return filtered_tokens


nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab)

phrases = [
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
    "not enough",
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
patterns = [nlp.make_doc(phrase) for phrase in phrases]
matcher.add("IMPORTANT_PHRASES", patterns)
def preserve_phrases(text):
    doc = nlp(text)
    matches = matcher(doc)
    preserved_text = text
    for match_id, start, end in matches:
        span = doc[start:end]
        # Replace with "down_the_drain"
        preserved_text = preserved_text.replace(span.text, "_".join(span.text.split()))
    return preserved_text


tokenizer = TweetTokenizer()
for text in text_list:
    text = text.strip()
    text = text.lower()

    text = re.sub(r"@\w+", '', text)  # remove @
    text = re.sub(r"#\w+", '', text)  # remove #
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)  # remove special characters
    text = re.sub(r'\s+', " ", text).strip()
    text = preserve_phrases(text)
    tokens = tokenizer.tokenize(text)
    tokens = contraction_filter(tokens)
    print(f"original: {tokens}")
    corrected_tokens = [correct_mispronounciation(token) for token in tokens]
    print(f"Corrected tokens: {corrected_tokens}")