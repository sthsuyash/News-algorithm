from snowballstemmer import NepaliStemmer


# Preprocess text function
def preprocess_text(cat_data, stop_words, punctuation_words):
    stemmer = NepaliStemmer()  # Initialize the Nepali stemmer
    new_cat = []
    noise = "1,2,3,4,5,6,7,8,9,0,०,१,२,३,४,५,६,७,८,९".split(",")

    for row in cat_data:
        words = row.strip().split(" ")
        nwords = ""

        for word in words:
            # Apply Nepali stemming to the word
            if word not in punctuation_words and word not in stop_words:
                word = stemmer.stemWord(word)

                is_noise = False
                for n in noise:
                    if n in word:
                        is_noise = True
                        break
                if not is_noise and len(word) > 1:
                    word = word.replace("(", "")
                    word = word.replace(")", "")
                    nwords += word + " "

        new_cat.append(nwords.strip())

    return new_cat
