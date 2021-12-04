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
Extra Credit: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""
import math
smoothing_constant = 1e-5

def viterbi_ec(train, test):
        '''
        input:  training data (list of sentences, with tags on the words)
                test data (list of sentences, no tags on the words)
        output: list of sentences with tags on the words
                E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        '''
        tags = {}
        start = {}
        pairs = {}
        countprev = {}
        counttag = {}
        hapwords = {}
        ly = {}
        ing = {}
        r = {}
        #note that start and end are considered tag as here
        for sentence in train:
                prev = "START"
                for data in sentence:
                        if data[1] == "END" or data[1] == "START":
                                continue;
                        if hapwords.get(data[0]) == None:
                                hapwords[data[0]] = data[1]
                        else:
                                hapwords[data[0]] = "REPEAT"
                        if data[0][-1] == 'r':
                                if r.get(data[0]) == None:
                                        r[data[0]] = data[1]
                                else:
                                        r[data[0]] = "REPEAT"
                        if data[0][-2:] == 'ly':
                                if ly.get(data[0]) == None:
                                        ly[data[0]] = data[1]
                                else:
                                        ly[data[0]] = "REPEAT"
                        if data[0][-3:] == "ing":
                                if ing.get(data[0]) == None:
                                        ing[data[0]] = data[1]
                                else:
                                        ing[data[0]] = "REPEAT"
                        if pairs.get(prev) == None:
                                pairs[prev] = {}
                                pairs[prev][data[1]] = 1;
                        else:
                                if pairs[prev].get(data[1]) == None:
                                        pairs[prev][data[1]] = 1
                                else:
                                        pairs[prev][data[1]] += 1
                        if countprev.get(prev) == None:
                                countprev[prev] = 1
                        else:
                                countprev[prev] += 1
                        prev = data[1]
                        if tags.get(data[1]) == None:
                                tags[data[1]] = {};
                                tags[data[1]][data[0]] = 1;
                        else:
                                if tags[data[1]].get(data[0]) == None:
                                        tags[data[1]][data[0]] = 1;
                                else:
                                        tags[data[1]][data[0]] += 1;
                        if counttag.get(data[1]) == None:
                                counttag[data[1]] = 1
                        else:
                                counttag[data[1]] += 1

        a = smoothing_constant;#soothing parameter
        # print(test)
        #note that Start and End don't count as tag
        probtag = {};
        probr = {};
        probly = {};
        probing = {};
        numhapword = 0
        numr = 0;
        numly = 0;
        numing = 0;
        for word in hapwords:
                if hapwords[word] != "REPEAT":
                        numhapword += 1
                        if probtag.get(hapwords[word]) == None:
                                probtag[hapwords[word]] = 1;
                        else:
                                probtag[hapwords[word]] += 1;
        for word in r:
                if r[word] != "REPEAT":
                        numr += 1
                        if probr.get(r[word]) == None:
                                probr[r[word]] = 1;
                        else:
                                probr[r[word]] += 1;
        for word in ly:
                if ly[word] != "REPEAT":
                        numly += 1
                        if probly.get(ly[word]) == None:
                                probly[ly[word]] = 1;
                        else:
                                probly[ly[word]] += 1;
        for word in ing:
                if ing[word] != "REPEAT":
                        numing += 1
                        if probing.get(ing[word]) == None:
                                probing[ing[word]] = 1;
                        else:
                                probing[ing[word]] += 1;
        for tag in tags:
                if probtag.get(tag) == None:
                        probtag[tag] = 1
                else:
                        probtag[tag] += 1
                if probr.get(tag) == None:
                        probr[tag] = 1
                else:
                        probr[tag] += 1
                if probing.get(tag) == None:
                        probing[tag] = 1
                else:
                        probing[tag] += 1
                if probly.get(tag) == None:
                        probly[tag] = 1
                else:
                        probly[tag] += 1

        for tag in probtag:
                probtag[tag] /= numhapword
        for tag in probing:
                probing[tag] /= numing
        for tag in probly:
                probly[tag] /= numly
        for tag in probr:
                probr[tag] /= numr

        # print(probtag)
        for prev in pairs:
                for tag in pairs[prev]:
                        pairs[prev][tag] = math.log(pairs[prev][tag] +  a)-math.log(countprev[prev] + a*(len(tags) + 1))
                pairs[prev]["UNKNOWN"] = math.log(0 +  a)-math.log(countprev[prev] + a*(len(tags) + 1))
        for tag in tags:
                for word in tags[tag]:
                        tags[tag][word] = math.log(tags[tag][word] + a*probtag[tag])-math.log(counttag[tag] + a*(len(tags[tag]) + 1))
                tags[tag]["UNKNOWN"] = math.log(0 + a*probtag[tag])-math.log(counttag[tag] + a*(len(tags[tag]) + 1))
                tags[tag]["UNr"] = math.log(0 + a*probr[tag])-math.log(counttag[tag] + a*(len(tags[tag]) + 1))
                tags[tag]["UNly"] = math.log(0 + a*probly[tag])-math.log(counttag[tag] + a*(len(tags[tag]) + 1))
                tags[tag]["UNing"] = math.log(0 + a*probing[tag])-math.log(counttag[tag] + a*(len(tags[tag]) + 1))
        print(probtag)
        output = []
        for sentence in test:
                # initialstate = pairs["START"]
                # firstword = sentence[1]
                # if (tag[firstword])
                # for starttag in initialstate:
                #         initialstate[starttag]*= tags[starttag][firstword]
                backptr = [] #an array of dictionary contains back pointer
                probab = [] #an array of dictionary contains node probability
                # probab.append(initialstate)
                for i in range(1,len(sentence) - 1):  #exclude START and END
                        word =sentence[i]
                        probdic = {}
                        backdic = {}
                        for tag in tags:
                                wordgiventag = 0
                                if tags[tag].get(word) == None:
                                        if word[-1] == "r":
                                                wordgiventag = tags[tag]["UNr"]
                                        elif word[-2:] == "ly":
                                                wordgiventag = tags[tag]["UNly"]
                                        elif word[-3:] == "ing":
                                                wordgiventag = tags[tag]["UNing"]
                                        else:
                                                wordgiventag = tags[tag]["UNKNOWN"]
                                else:
                                        wordgiventag = tags[tag][word]
                                maxprob = -math.inf
                                bkptr = ""
                                prevstate = {"START": math.log(1)}
                                if (len(probab) != 0) :
                                        prevstate = probab[-1]
                                for prevtag in prevstate:
                                        taggivenprev = 0
                                        if pairs[prevtag].get(tag) == None:
                                                taggivenprev = pairs[prevtag]["UNKNOWN"]
                                        else:
                                                taggivenprev = pairs[prevtag][tag]
                                        if taggivenprev+wordgiventag+prevstate[prevtag] > maxprob:
                                                bkptr = prevtag
                                                maxprob = taggivenprev+wordgiventag+prevstate[prevtag]
                                probdic[tag] = maxprob
                                backdic[tag] = bkptr        
                        probab.append(probdic)
                        backptr.append(backdic)
                finaltag = probab[-1]
                final = ""
                maxfinal = -math.inf
                for ft in finaltag:
                        if finaltag[ft] > maxfinal:
                                maxfinal = finaltag[ft]
                                final = ft
                        # endgivenft = 0
                        # if pairs[ft].get("END") == None:
                        #         endgivenft = pairs[ft]["UNKNOWN"]
                        # else:
                        #         endgivenft = pairs[ft]["END"]
                        # if endgivenft+np.log(1)+finaltag[ft] > maxfinal:
                        #         final = ft
                        #         maxfinal = endgivenft
                sentag = []
                sentag.append(tuple(["END","END"]))
                lastword = sentence[len(sentence) - 2]
                lasttag = final
                sentag.insert(0, tuple([lastword, lasttag]))
                for i in range(len(sentence) - 3):
                        word = sentence[len(sentence) - 3 - i]
                        tag = backptr[len(backptr) - 1 - i][lasttag]
                        lasttag = tag
                        sentag.insert(0, tuple([word, tag]))
                sentag.insert(0,tuple(["START","START"]))
                output.append(sentag)
        return output
