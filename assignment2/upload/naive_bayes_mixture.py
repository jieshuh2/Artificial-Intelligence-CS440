# naive_bayes_mixture.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import math



def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda,unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """

    # TODO: Write your code here
    # return predicted labels of development set
    
    #train unigram
    dicham = {}
    dicspam = {}
    numham = 0
    numspam = 0
    for index in range(len(train_set)):
        ham = False
        if train_labels[index] == 1 :
            ham = True
        else:
            ham = False
            
        #store all words in that emal. 
        for word in train_set[index]:
            if ham:
                if dicham.get(word) == None:
                    dicham[word] = 1;
                else:
                    dicham[word] += 1;
                numham += 1
            else:
                if dicspam.get(word) == None:
                    dicspam[word] = 1;
                else:
                    dicspam[word] += 1;
                numspam += 1
    #likelihood need smooth #log(P)
    P_word_over_ham = {};
    for word in dicham:
        P_word_over_ham[word] = math.log(dicham[word] + unigram_smoothing_parameter)- math.log(numham + unigram_smoothing_parameter * len(dicham))
    P_word_over_ham["none"] = math.log(unigram_smoothing_parameter) - math.log(numham + unigram_smoothing_parameter * len(dicham))
    P_word_over_spam = {};
    for word in dicspam:
        P_word_over_spam[word] = math.log(dicspam[word] + unigram_smoothing_parameter) - math.log(numspam + unigram_smoothing_parameter * len(dicspam))
    P_word_over_spam["none"] = math.log(unigram_smoothing_parameter) - math.log(numspam + unigram_smoothing_parameter*len(dicspam))

    #train binary
    biham = {}
    bispam = {}
    numbiham = 0
    numbispam = 0
    for i in range(len(train_set)):
        isham = False
        if train_labels[i] == 1:
            isham = True
        else:
            isham = False

        for j in range(len(train_set[i])  - 1):
            word = train_set[i][j] + " " + train_set[i][j + 1]
            if isham:
                if biham.get(word) == None:
                    biham[word] = 1;
                else:
                    biham[word] += 1;
                numbiham += 1
            else:
                if bispam.get(word) == None:
                    bispam[word] = 1;
                else:
                    bispam[word] += 1;
                numbispam += 1
    P_bi_over_ham = {};
    for word in biham:
        P_bi_over_ham[word] = math.log(biham[word] + bigram_smoothing_parameter) - math.log(numbiham + bigram_smoothing_parameter*len(biham))
    P_bi_over_ham["none"] = math.log(bigram_smoothing_parameter) - math.log(numbiham + bigram_smoothing_parameter*len(biham))

    P_bi_over_spam = {};
    for word in bispam:
        P_bi_over_spam[word] = math.log(bispam[word] + bigram_smoothing_parameter) - math.log(numbispam + bigram_smoothing_parameter*len(bispam))
    P_bi_over_spam["none"] = math.log(bigram_smoothing_parameter) - math.log(numbispam + bigram_smoothing_parameter*len(bispam))

    #development
    P_posterior_ham = []
    P_posterior_spam = []
    for email in dev_set:
        p_ham = (math.log(pos_prior))*(1 - bigram_lambda) + (math.log(pos_prior))*(bigram_lambda)
        p_spam = (math.log(1- pos_prior))*(1 - bigram_lambda) + (math.log(1 - pos_prior))*(bigram_lambda)
        for word in email:
            p_ham_word = P_word_over_ham.get(word)
            if p_ham_word == None:
                p_ham_word =  P_word_over_ham["none"]
            p_spam_word = P_word_over_spam.get(word)
            if p_spam_word == None:
                p_spam_word =  P_word_over_spam["none"]
            p_ham += (1 - bigram_lambda)*p_ham_word
            p_spam += (1 - bigram_lambda)*p_spam_word

        for j in range(len(email) - 1):
            word = email[j] + " " + email[j + 1]
            bi_ham_word = P_bi_over_ham.get(word)
            if bi_ham_word == None:
                bi_ham_word =  P_bi_over_ham["none"]
            bi_spam_word = P_bi_over_spam.get(word)
            if bi_spam_word == None:
                bi_spam_word =  P_bi_over_spam["none"]
            p_ham += bigram_lambda * bi_ham_word
            p_spam += bigram_lambda * bi_spam_word
        P_posterior_ham.append(p_ham)
        P_posterior_spam.append(p_spam)
    typepredict = []
    for i in range(len(dev_set)):
        predict = 0
        if P_posterior_ham[i] >= P_posterior_spam[i]:
            predict = 1
        else:
            pretict = 0
        typepredict.append(predict)
    return typepredict