import numpy as np
import math
import nltk
from math import log
from operator import itemgetter
from nltk.corpus import treebank
import argparse

class HybridValue:
    def __init__(self):
        self._counter = 0
        self._nested = None  # Will become a dict if used as a mapping

    def _ensure_nested(self):
        if self._nested is None:
            self._nested = {}

    def __iadd__(self, other):
        if self._nested is not None:
            raise TypeError("This HybridValue has been promoted to a nested mapping and cannot be incremented directly.")
        self._counter += other
        return self

    def __getitem__(self, key):
        # When indexing, promote to mapping mode if not already.
        self._ensure_nested()
        if key not in self._nested:
            self._nested[key] = HybridValue()
        return self._nested[key]

    def __setitem__(self, key, value):
        # When assigning, promote to mapping mode.
        self._ensure_nested()
        # If assigning an int, store it inside a HybridValue.
        if isinstance(value, int):
            if key not in self._nested:
                self._nested[key] = HybridValue()
            self._nested[key]._counter = value
        elif isinstance(value, HybridValue):
            self._nested[key] = value
        else:
            # For any other type, store it directly.
            self._nested[key] = value

    def __repr__(self):
        # When not promoted, show the counter.
        if self._nested is None:
            return repr(self._counter)
        else:
            return repr(self._nested)


class SimpleCFD(dict):
    def __getitem__(self, key):
        if key not in self:
            self[key] = HybridValue()
        return dict.__getitem__(self, key)

class TnTTagger:
    def __init__(self):
        self.w_t = SimpleCFD()
        self.uni_t = SimpleCFD()
        self.bi_t = SimpleCFD()
        self.tri_t = SimpleCFD()
        self._EOS = SimpleCFD()
        self.size_ = 0
        self._N = 1000

        # Initialize lambda values
        self.l1 = 0
        self.l2 = 0
        self.l3 = 0

    def __populate__(self, trainset):
        for sent in trainset:
            
            current_states = [('BOS', False), ('BOS', False)]
            for word, tag in sent:
                self.size_ += 1
                _uP = word[0].isupper()

                tag = (tag, _uP)

                self.w_t[word][tag] += 1
                self.uni_t[tag] += 1
                self.bi_t[current_states[1]][tag] += 1
                self.tri_t[tuple(current_states)][tag] += 1

                current_states.append((tag))
                current_states.pop(0)

            self._EOS[tag][('EOS', False)] += 1
            self.size_ += 1

        self.__lambda__()

    def __lambda__(self):
        for t1t2 in self.tri_t.keys():
            for t3 in self.tri_t[t1t2]._nested.keys():
                if self.tri_t[t1t2][t3]._counter > 0:
                    f_tri = (
                        (self.tri_t[t1t2][t3]._counter - 1) / (self.bi_t[t1t2[0]][t1t2[1]]._counter - 1 or math.inf)
                    )

                    f_bi = (
                        (self.bi_t[t1t2[1]][t3]._counter - 1) / (self.uni_t[t1t2[1]]._counter - 1 or math.inf)
                    )

                    f_uni = (
                        (self.uni_t[t3]._counter - 1) / (self.size_ - 1 or math.inf)
                    )

                    if f_tri == max(f_tri, f_bi, f_uni):
                        self.l3 += self.tri_t[t1t2][t3]._counter
                    
                    if f_bi == max(f_tri, f_bi, f_uni):
                        self.l2 += self.tri_t[t1t2][t3]._counter

                    if f_uni == max(f_tri, f_bi, f_uni):
                        self.l1 += self.tri_t[t1t2][t3]._counter
        
        net_lmbda = np.sqrt(sum(l**2 for l in [self.l1, self.l2, self.l3]))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        net_lmbda = self.l1 + self.l2 + self.l3

        self.l1 = self.l1/net_lmbda
        self.l2 = self.l2/net_lmbda
        self.l3 = self.l3/net_lmbda
    
    def __tagdata__(self, sentences):
        tags = []
        for sent in sentences:
            pred_tags = self.tag_sent(list(sent))
            tags.append(pred_tags)
        
        return tags
    
    def tag_sent(self, sent):

        current_state = [(["BOS", "BOS"], 0.0)]
        tags = self.beamSearchTagger(sent, current_state)

        res = []
        for i in range(len(sent)):
            (t, C) = tags[i + 2]
            res.append((sent[i], t))

        return res

    def beamSearchTagger(self, sent, current_states):

        if sent == []:
            (h, logp) = current_states[0]
            return h

        word = sent[0]
        sent = sent[1:]
        new_states = []

        C = False
        if word[0].isupper():
            C = True

        if word in self.w_t:

            for history, curr_sent_logprob in current_states:
                for t in self.w_t[word]._nested.keys():
                    tag = (t, C)
                    p_uni = self.uni_t[tag]._counter/self.size_
                    p_bi = self.bi_t[history[-1]][tag]._counter/(self.uni_t[history[-1]]._counter or math.inf)
                    p_tri = self.tri_t[tuple(history[-2:])][tag]._counter/(self.bi_t[history[-2]][history[-1]]._counter or math.inf)
                    p_w = self.w_t[word][t]._counter /(self.uni_t[tag]._counter or math.inf)
                    p = self.l1 * p_uni + self.l2 * p_bi + self.l3 * p_tri
                    try:
                        p2 = log(p, 2) + log(p_w, 2)
                    except:
                        p2 = 0.1
                    new_states.append((history + [tag], curr_sent_logprob + p2))

        else:
            
            tag = ("Unk", C)

            for history, _ in current_states:
                history.append(tag)

            new_states = current_states


        new_states.sort(reverse=True, key=itemgetter(1))

        if len(new_states) > self._N:
            new_states = new_states[: self._N]

        return self.beamSearchTagger(sent, new_states)


if __name__ == '__main__':
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Process a string input.")

    # Add an argument for the data
    parser.add_argument("--sentence", type=str, required=False, help="Input sentence data for Part-of-Speech Tagging")
    parser.add_argument("--traindata", type=str, required=False, help="Input dataset for training TnTTagger")

    # Parse the arguments
    args = parser.parse_args()

    # load the POS tagged data
    traindata = args.traindata if args.traindata else 'treebank'
    nltk.download(traindata)
    data = treebank.tagged_sents()

    # Initialize the tagger and populate it
    tagger = TnTTagger()    
    tagger.__populate__(data)

    # Run the TnT tagger
    test_sent = "This is a test sentence to generate part of speech tags"
    sentence = args.sentence if args.sentence else test_sent
    sentence = nltk.word_tokenize(sentence)
    tags = tagger.tag_sent(sentence)

    res = []
    for w, t in tags:
        res.append((w, t[0]))
    print(f"\n\nSentence: {sentence}\nPOS Tags: {res}")
