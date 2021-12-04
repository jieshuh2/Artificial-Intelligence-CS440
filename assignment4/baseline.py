# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
        '''
        input:  training data (list of sentences, with tags on the words)
                test data (list of sentences, no tags on the words)
        output: list of sentences, each sentence is a list of (word,tag) pairs.
                E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        '''
        pairs = {}
        # print(train[0])
        tagnum = {}
        for sentence in train:
                for data in sentence:
                        if tagnum.get(data[1]) == None:
                                tagnum[data[1]] = 1;
                        else :
                                tagnum[data[1]] += 1;
                        if pairs.get(data[0]) == None:
                                pairs[data[0]] = {};
                                pairs[data[0]][data[1]] = 1;
                        else:
                                if pairs[data[0]].get(data[1]) == None:
                                        pairs[data[0]][data[1]] = 1;
                                else:
                                        pairs[data[0]][data[1]] += 1;
        tags = {}
        for word in pairs:
                bestag = ""
                num = 0
                for tag in pairs[word]:
                        if pairs[word][tag] > num:
                                num = pairs[word][tag]
                                bestag = tag
                
                tags[word] = bestag
        numtag = 0
        best = ""
        for tagi in tagnum:
                if tagnum[tagi] > numtag:
                        numtag = tagnum[tagi]
                        best = tagi
        output = []
        for sentence in test:
                sent = []
                for word in sentence:
                        if tags.get(word) == None:
                                sent.append(tuple([word, best]))
                        else:
                                sent.append(tuple([word, tags[word]]))
                output.append(sent)
        return output