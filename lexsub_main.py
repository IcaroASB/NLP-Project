#!/usr/bin/env python
# This code was complete as part of a class project and is not of my full authorship
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context
import string


from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import numpy as np
import tensorflow as tf

import gensim
from typing import List
import transformers 

from typing import List

def tokenize(s): 
    """
    A naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> set:

    syns = set()

    # not much creativity here
    # just check all words in synsets for the lemna and every word
    # that is not the lemma itself is a synonym

    for synset in wn.synsets(lemma, pos):

        for lemma_synset in synset.lemmas(): 
            syn = lemma_synset.name().replace('_', ' ')

            if syn != lemma:
                syns.add(syn)

    return syns

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(ctx: Context) -> str:

    cands = get_candidates(ctx.lemma, ctx.pos)

    top_cand, max_freq = None, 0

    # then we get the frequency of each canidate
    for cand in cands:
        f = sum(lm.count() for lm in wn.lemmas(cand.replace(' ', '_')))
        if f > max_freq:
            max_freq, top_cand = f, cand

    return top_cand

def wn_simple_lesk_predictor(ctx: Context) -> str:

    # first, we rmv stopwrods from context to get meaningful words
    ctx_wds = set(tokenize(' '.join(ctx.left_context + ctx.right_context))) - set(stopwords.words('english'))

    def calc_ovl(syn, wds):
        def_wds = set(tokenize(syn.definition()))

        def_wds.update(tkn for ex in syn.examples() for tkn in tokenize(ex))

        def_wds.update(tkn for hypernym in syn.hypernyms() for tkn in tokenize(hypernym.definition()))

        def_wds.update(tkn for hypernym in syn.hypernyms() for ex in hypernym.examples() for tkn in tokenize(ex))

        # then we ccal overlap of word sets to find besst fit
        return len(def_wds & wds)

    b_sense, h_overlap = max(((s, calc_ovl(s, ctx_wds)) for s in wn.synsets(ctx.lemma, ctx.pos)), key=lambda x: x[1], default=(None, 0))

    syn = (lambda lm: lm.name().replace('_', ' '))(
        max(b_sense.lemmas(), key=lambda x: x.count(), default=None)) if b_sense else None
    
    return syn if syn and syn != ctx.lemma else wn_frequency_predictor(ctx)
    

class Word2VecSubst(object):
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self, context: Context) -> str:
        cands = get_candidates(context.lemma, context.pos)

        # here we calc similarity among candidtes
        sim = [(cand, self.model.similarity(context.lemma, cand.replace(' ', '_')))
               
                        for cand in cands
                        if cand in self.model]

        # find best candidate
        # tried to write max in a simple way that does not break if there are no cands
        # did not break in testing with the words provided
        best_cand = max(sim, key=lambda x: x[1], default=(None, -1))[0]

        return best_cand

class BertPredictor(object):
    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def get_masked_input(self, ctx):

        # we start by creating a massk for predict
        sentence = f"{' '.join(ctx.left_context)} [MASK] {' '.join(ctx.right_context)}"
        return self.tokenizer(sentence, return_tensors='tf')

    def get_mask_idx(self, input_ids):
        return tf.where(input_ids == self.tokenizer.mask_token_id)[0, 1]

    def find_best_cand(self, cands: List[str], preds, mask_idx):

        best_score, best_cand = -float('inf'), None

        for word in cands:

            word_id = self.tokenizer.convert_tokens_to_ids(word)
            word_score = preds[0, mask_idx, word_id].numpy()

            if word_score > best_score:
                best_score, best_cand = word_score, word

        return best_cand

    def predict(self, ctx) -> str:

        # we get cand words
        cands = get_candidates(ctx.lemma, ctx.pos)
        input_data = self.get_masked_input(ctx)
        mask_idx = self.get_mask_idx(input_data['input_ids'])

        # proceed to gt predcts

        preds = self.model(input_data).logits
        return self.find_best_cand(cands, preds, mask_idx)

class Method6Predictor(object):

    # the idea behind this method is simple: it uses both Word2Vec and WordNet
    # it sorts all cadidates from Word2Vec and select the top 3 instead of the first
    # if the candidate given by WordNet (frequency) method is among these 3, this is the choosen word
    # otherwise we use the mhighest among the 3 candiates

    def __init__(self, w2v_filename):
        self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_filename, binary=True)

    def predict(self, context: Context) -> str:

        cands = get_candidates(context.lemma, context.pos)
        w2v_scores = self._get_word2vec_scores(context, cands)

        top_w2v_cands = sorted(w2v_scores, key=w2v_scores.get, reverse=True)[:3]

        freq_cand = wn_frequency_predictor(context)

        if freq_cand in top_w2v_cands:
            return freq_cand
        else:
            return top_w2v_cands[0]

    # helper to rank candidates fron word2vec
    def _get_word2vec_scores(self, context, cands):

        scores = {}

        for cand in cands:

            if cand in self.word2vec_model:

                try:
                    score = self.word2vec_model.similarity(context.lemma, cand.replace(' ', '_'))
                    scores[cand] = score

                except KeyError:
                    pass
                
        return scores

if __name__=="__main__":
    predictor = BertPredictor()

    for context in read_lexsub_xml(sys.argv[1]):
        prediction = predictor.predict(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
