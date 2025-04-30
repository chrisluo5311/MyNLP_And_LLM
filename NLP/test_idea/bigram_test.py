from symspellpy import SymSpell


text_list = ["THAT one didn't work either", "I'm very disappointed with my decision", "Absolutel junk",
            "It dit not work most of the time with my Nokia 5320.", "down the drain", "out of hand",
            "In short - this was a monumental waste of time and energy and I would not recommend anyone to EVER see this film.",
            "My sashimi was poor quality being soggy and tasteless.",
            "Best fish i've ever taste",
            "about a hour ago I ordered a pizza from this place and it was the worst pizza I have ever had.",
            "not god product","it dit not wrk"]


symSpell2 = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
symSpell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
symSpell2.load_bigram_dictionary("./frequency_bigramdictionary_en_243_342.txt", term_index=0, count_index=2, separator=' ')
symSpell.load_dictionary("./frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)
for each_txt in text_list:
    print(f"original: {each_txt}")
    possible_answer2 = symSpell2.lookup_compound(each_txt, max_edit_distance=2)
    for suggestion in possible_answer2:
        print(f"[Bigram] Corrected: {suggestion.term}, Confidence: {suggestion.distance}")
    if possible_answer2:
        each_txt = possible_answer2[0].term

    possible_answer1 = symSpell.lookup_compound(each_txt, max_edit_distance=2)
    for suggestion in possible_answer1:
        print(f"[Unigram] Corrected: {suggestion.term}, Confidence: {suggestion.distance}")

    print()
