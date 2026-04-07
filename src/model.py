'''
These are some fuctions I use for the training model
- naive_bayes(): for counting word frequencies and compute prior probabilites
- predict(): classify a message as SPAM or HAM using log-space Naive Bayes scoring
'''
import math
from utils import preprocess
def naive_bayes(dataset):
    total = len(dataset)
    spam_count = 0
    for item in dataset:
        if item[1] == "spam":
            spam_count+=1
    ham_count = total - spam_count

    p_spam = spam_count / total
    p_ham = ham_count / total

    spam_words = {}
    ham_words = {}
    vocabulary = set()

    for message, label in dataset:
        words = preprocess(message)
        for word in words:
            vocabulary.add(word)
            if label == "spam":
                spam_words[word] = spam_words.get(word, 0) + 1
            else:
                ham_words[word] = ham_words.get(word, 0) + 1
                
    return {
        "p_spam": p_spam,
        "p_ham": p_ham,
        "spam_words": spam_words,
        "ham_words": ham_words,
        "total_spam_words": sum(spam_words.values()),
        "total_ham_words": sum(ham_words.values()),
        "vocab_size": len(vocabulary)
    }

def predict(new_message, model):
    words = preprocess(new_message)
 
    # GIBBERISH FILTER 
    # If every token is unknown, default scoring would just compare priors
    # (ham wins by default). Unknown token streams are more likely spam.
    knownwords = sum(1 for w in words if w in model["spam_words"] or w in model["ham_words"])
    if len(words) > 0 and knownwords == 0: 
        return "SPAM (Unknown words)"
 
    score_spam = math.log(model["p_spam"])
    score_ham  = math.log(model["p_ham"])
 
    for word in words:
        if word not in model["spam_words"] and word not in model["ham_words"]:
            continue
 
        p_word_given_spam = (model["spam_words"].get(word, 0) + 1) / (model["total_spam_words"] + model["vocab_size"])
        score_spam += math.log(p_word_given_spam)
 
        p_word_given_ham  = (model["ham_words"].get(word,  0) + 1) / (model["total_ham_words"]  + model["vocab_size"])
        score_ham  += math.log(p_word_given_ham)
 
    return "SPAM" if score_spam > score_ham else "HAM"