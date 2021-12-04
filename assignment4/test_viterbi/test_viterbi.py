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
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
This file should not be submitted - it is only meant to test your implementation of the Viterbi algorithm. 
"""
from utils import read_files, get_nested_dictionaries
import math
smoothing_constant = 1e-10

def main():
    test, emission, transition, output = read_files()
    emission, transition = get_nested_dictionaries(emission, transition)
    initial = transition["S"]
    prediction = []
    
    """WRITE YOUR VITERBI IMPLEMENTATION HERE"""
#     print(transition["VB"])
    backptr = [] #an array of dictionary contains back pointer
    probab = [] #an array of dictionary contains noderobability
    for tag in initial:
        initial[tag] *= emission[tag][test[0][0]]
    probab.append(initial)
    for i in range(1,len(test[0])):  #exclude START and END
            word = test[0][i]
            probdic = {}
            backdic = {}
            for tag in emission:
                    wordgiventag = math.log(emission[tag][word])
                    maxprob = -math.inf
                    bkptr = ""
                    prevstate = probab[-1]
                    for prevtag in prevstate:
                            taggivenprev = math.log(transition[prevtag][tag])
                            probprev = prevstate[prevtag]
                            if taggivenprev+wordgiventag+prevstate[prevtag] > maxprob:
                                    bkptr = prevtag
                                    maxprob = taggivenprev+wordgiventag+prevstate[prevtag]
                    probdic[tag] = maxprob
                    backdic[tag] = bkptr        
            probab.append(probdic)
            backptr.append(backdic)
    print(backptr)
    print(len(test[0]))
    print(len(backptr))
    finaltag = probab[-1]
    final = ""
    maxfinal = -math.inf
    for tag in finaltag:
        if finaltag[tag] > maxfinal:
            maxfinal = finaltag[tag]
            final = tag
    sentag = []
    # sentag.append(tuple(["END","END"]))
    lastword = test[0][-1]
    lasttag = final
    sentag.insert(0, tuple([lastword, lasttag]))
    for i in range(len(test[0]) - 1):
            word = test[0][len(test[0]) - 2 - i]
            tag = backptr[len(backptr) - 1 - i][lasttag]
            lasttag = tag
            sentag.insert(0, tuple([word, tag]))
    # sentag.insert(0,tuple(["START","START"]))
    prediction = sentag

    print('Your Output is:',prediction,'\n Expected Output is:',output)
    


if __name__=="__main__":
    main()