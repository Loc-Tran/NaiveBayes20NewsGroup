# By Phuoc Loc Tran
# Assuming filenames and path names remain the same from the dataset 20news-bydate.tar.gz
# and these files are in the same directory as this python code
# http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz

import os
import re
from decimal import *

def getVocabulary(V):
    print("Getting vocabulary...\n")
    vocabulary = set()
    cardinality_of_examples = 0
    for v in V:
        print("Processing folder " + v + "...\n")
        textfile_names = os.listdir("20news-bydate-train" + "/" + v)
        cardinality_of_examples += len(textfile_names)
        for textfile in textfile_names:
            try:
                with open("20news-bydate-train" + "/" + v + "/" + textfile) as f:
                    print("Processing " + "20news-bydate-train" + "/" + v + "/" + textfile + "...")
                    for line in f:
                        for word in re.findall(r"[A-Za-z]+", line):
                            vocabulary.add(word)
            except UnicodeDecodeError:
                with open("20news-bydate-train" + "/" + v + "/" + textfile, encoding="iso-8859-15") as f:
                    print("Processing " + "20news-bydate-train" + "/" + v + "/" + textfile + "using encoding iso-8859-15...")
                    for line in f:
                        for word in re.findall(r"[A-Za-z]+", line):
                            vocabulary.add(word)                
    return [vocabulary, cardinality_of_examples]        

def initialise_pw_given_v(vocabulary, V):
    print("Initialising pw_given_v...\n")
    pw_given_v = {}
    for v in V:
        pw_given_v[v] = {}
        for word in vocabulary:
            pw_given_v[v][word] = 0
    return pw_given_v

def initialise_count(vocabulary, V): # count[v][wordj] = number of word in v
    print("Initialising count...\n")
    count = {}
    for v in V:
        count[v] = {}
        for word in vocabulary:
            count[v][word] = 0
    return count

def learn_naive_bayes_text(cardinality_of_examples, vocabulary, V):
    print("Learning training set...\n")
    pw_given_v = initialise_pw_given_v(vocabulary, V)
    Pv = {}
    count = initialise_count(vocabulary, V)   
    for v in V:
        print("Processing value" + v + "...\n")
        textfile_names = os.listdir("20news-bydate-train" + "/" + v)
        cardinality_of_docs = len(textfile_names)
        Pv[v] = Decimal(cardinality_of_docs / cardinality_of_examples)
        number_of_distinct_word_positions = 0
        for textfile in textfile_names:
            try:
                with open("20news-bydate-train" + "/" + v + "/" + textfile) as f:       
                    print("Counting word positions " + "20news-bydate-train" + "/" + v + "/" + textfile + "...")       
                    for line in f:
                        regex = re.findall(r"[A-Za-z]+", line)
                        number_of_distinct_word_positions += len(regex)
                        for word in regex:
                            print("Counting word " + word + " in 20news-bydate-train" + "/" + v + "/" + textfile + "...")    
                            count[v][word] += 1
            except UnicodeDecodeError:
                with open("20news-bydate-train" + "/" + v + "/" + textfile, encoding="iso-8859-15") as f:   
                    print("Counting word positions " + "20news-bydate-train" + "/" + v + "/" + textfile + "using encoding iso-8859-15...")             
                    for line in f:
                        regex = re.findall(r"[A-Za-z]+", line)
                        number_of_distinct_word_positions += len(regex)
                        for word in regex:
                            print("Counting word " + word + "in 20news-bydate-train" + "/" + v + "/" + textfile + "using encoding iso-8859-15...") 
                            count[v][word] += 1          
        for vocab_word in vocabulary: #vocab_word == wk
            pw_given_v[v][vocab_word] = Decimal((count[v][vocab_word] + 1) / (number_of_distinct_word_positions + len(vocabulary)))
            print("Storing pw_given_v[{}][{}] as {}...".format(v, vocab_word, pw_given_v[v][vocab_word]))
    return Pv, pw_given_v, count

def classify_naives_bayes_text(path_to_document, pw_given_v, vocabulary, V, Pv):
    print("Classifying textfile at " + path_to_document + "...")
    positions = []
    try:
        with open(path_to_document) as f:
            for line in f:
                for word in re.findall(r"[A-Za-z]+", line):
                    if word in vocabulary:
                        positions.append(word)
    except UnicodeDecodeError:
        with open(path_to_document, encoding="iso-8859-15") as f:
            for line in f:
                for word in re.findall(r"[A-Za-z]+", line):
                    if word in vocabulary:
                        positions.append(word)
    results = {}
    given_p_for_v = 0
    for v in V:
        given_p_for_v = Pv[v]
        for word in positions:
            given_p_for_v *= pw_given_v[v][word]
        results[given_p_for_v] = v
    arg_max = max(results.keys())
    return results[arg_max]

def main():
    V = os.listdir("20news-bydate-train")
    vocabulary, cardinality_of_examples = getVocabulary(V)
    Pv, pw_given_v, count = learn_naive_bayes_text(cardinality_of_examples, vocabulary, V)
    number_of_documents_in_test = 0
    documents_classified_correctly = 0
    number_of_documents_classified_as_v = {}
    number_of_documents_classified_as_v_classified_correctly = {}
    for v in V:
        number_of_documents_classified_as_v_classified_correctly[v] = 0
        number_of_documents_classified_as_v[v] = 0
    for v in V:
        _ = os.listdir("20news-bydate-test/" + v)
        number_of_documents_in_test += len(_)
        number_of_documents_classified_as_v[v] += len(_)

    v_given_by_classifer = ""
    print("Now beginning cross validation...\n")
    for v in V:
        textfile_names = os.listdir("20news-bydate-test/" + v)
        for textfile in textfile_names:
            v_given_by_classifer = classify_naives_bayes_text("20news-bydate-test/" + v + "/" + textfile, pw_given_v, vocabulary, V, Pv)
            if(v == v_given_by_classifer):
                print("Document {} classified as {} and is correct\n".format(textfile, v_given_by_classifer))
                documents_classified_correctly += 1
                number_of_documents_classified_as_v_classified_correctly[v] += 1
            else:
                print("Document {} classified as {} and is incorrect. Classification should be {}.\n".format(textfile, v_given_by_classifer, v))
    print("The top 5 words that were classified given {} are:".format(v))
    for v in V:
        sum_of_v_documents = 0
        count_of_words_given_v = count[v]
        for word in vocabulary:
            sum_of_v_documents += count_of_words_given_v[word]
        for v in V:
            top_5 = sorted(count_of_words_given_v.items(), key=lambda x:-x[1])[:5]
            for k, value in top_5:
                print("{} with a count of {} and comprising {}% of v's counted words.\n".format(k, value, (value/sum_of_v_documents)*100))
    print("There are {} documents in the training set.\n".format(cardinality_of_examples))
    print("There are {} documents in the testing set.\n".format(number_of_documents_in_test))
    print("{} documents were classified correctly using the naive bayes text classifer.\n".format(documents_classified_correctly))
    print("The accuracy rate is {}%.".format((documents_classified_correctly/number_of_documents_in_test) * 100))
    for v in V:
        print("For {}, there were {} test documents and {} were classified correctly. The accuracy rate for {} is {}%.".format(v, number_of_documents_classified_as_v[v], number_of_documents_classified_as_v_classified_correctly[v], v, (number_of_documents_classified_as_v_classified_correctly[v]/number_of_documents_classified_as_v[v])*100))
    
if __name__ == "__main__":
    main()
