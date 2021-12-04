# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for Part 1 of this MP. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import math


def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - positive prior probability (between 0 and 1)
    """
    # TODO: Write your code here
    # return predicted labels of development set
    

    #training phase
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
        P_word_over_ham[word] = math.log(dicham[word] + smoothing_parameter)- math.log(numham + smoothing_parameter * len(dicham))
    P_word_over_ham["none"] = math.log(smoothing_parameter) - math.log(numham + smoothing_parameter * len(dicham))
    P_word_over_spam = {};
    for word in dicspam:
        P_word_over_spam[word] = math.log(dicspam[word] + smoothing_parameter) - math.log(numspam + smoothing_parameter * len(dicspam))
    P_word_over_spam["none"] = math.log(smoothing_parameter) - math.log(numspam + smoothing_parameter*len(dicspam))
    #developing state
    P_posterior_ham = []
    P_posterior_spam = []
    for email in dev_set:
        p_ham = math.log(pos_prior)
        p_spam = math.log(1- pos_prior)
        for word in email:
            p_ham_word = P_word_over_ham.get(word)
            if p_ham_word == None:
                p_ham_word =  P_word_over_ham["none"]
            p_spam_word = P_word_over_spam.get(word)
            if p_spam_word == None:
                p_spam_word =  P_word_over_spam["none"]
            p_ham += p_ham_word
            p_spam += p_spam_word
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
    