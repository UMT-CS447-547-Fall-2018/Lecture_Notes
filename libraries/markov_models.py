from __future__ import division

import numpy as np
import string
from collections import Counter
import re
import json
import unicodedata

class NaiveBayesModel(object):
    """
    Implements a naive Bayes model for text generation and classification

    Attributes:
        terminal_characters: a set of punctuation that shouldn't fall at the
                             beginning of a sentence, but should end one.

    Methods:
        build_transition_matrices: builds a dictionary of word frequency
        generate_phrase: generates a phrase by randomly sampling from 
                         word frequencies
        compute_log_likelihood: computes the logarithm of the likelihood
                                for a specified phrase

    """

    terminal_characters = ['.','?','!']

    def __init__(self,sequence):
        """
        sequence: an ordered list of words corresponding to the training set
        """
        self.order=0
        self.sequence = sequence
        self.sequence_length = len(sequence)
        self.transitions = [{}]
        for i in range(self.order):
            self.transitions.append({})
        
    def build_transition_matrices(self):
        """
        Builds a dictionary of word probabilities
        """
        for i in range(self.sequence_length):
            word = self.sequence[i]
            if word in self.transitions[0]:
                self.transitions[0][word] += 1
            else:
                self.transitions[0][word] = 1

        # Convert counts to probabilities
        transition_sum = float(sum(self.transitions[0].values()))
        for k,v in self.transitions[0].items():
            self.transitions[0][k] = v/transition_sum

    def generate_phrase(self):
        """
        Take a random sample from the probability distribution.  Terminate
        when a period, question mark, or exclamation point is drawn.
        """
        w_i = np.random.choice(self.transitions[0].keys(),replace=True,p=self.transitions[0].values())
        phrase = w_i + ' '
        while w_i not in self.terminal_characters:
            w_i = np.random.choice(self.transitions[0].keys(),replace=True,p=self.transitions[0].values())
            phrase += w_i + ' '
        return phrase

    def compute_log_likelihood(self,phrase,lamda=0.0,unknown_probability=1e-5):
        """
        Return the log-probability of a given phrase (entered as a string)
        lambda: regularization factor for unseen transitions
        unknown_probability: probability mass of a word not in the dictionary.
        """
        words_in = phrase.split()
        log_prob = 0
        for word in words_in: 
            log_prob += np.log(self.transitions[0][word])
        return log_prob


class FirstOrderMarkovModel(object):
    """
    Implements a bigram model for text generation and classification

    Attributes:
        terminal_characters: a set of punctuation that shouldn't fall at the
                             beginning of a sentence, but should end one.

    Methods:
        build_transition_matrices: builds a dictionary of word frequency
        generate_phrase: generates a phrase by randomly sampling from 
                         word frequencies
        compute_log_likelihood: computes the logarithm of the likelihood
                                for a specified phrase

    """


    terminal_characters = ['.','?','!']

    def __init__(self,sequence):
        """
        sequence: an ordered list of words corresponding to the training set
        """

        self.order=1
        self.sequence = sequence
        self.sequence_length = len(sequence)
        self.transitions = [{}]
        for i in range(self.order):
            self.transitions.append({})
        

    def build_transition_matrices(self):
        """
        Builds a set of nested dictionaries of word probabilities
        """

        for i in range(self.sequence_length):
            word = self.sequence[i]
            if word in self.transitions[0]:
                self.transitions[0][word] += 1
            else:
                self.transitions[0][word] = 1

        transition_sum = float(sum(self.transitions[0].values()))
        for k,v in self.transitions[0].items():
            self.transitions[0][k] = v/transition_sum

        for i in range(self.sequence_length-1):
            word = self.sequence[i]
            next_word = self.sequence[i+1]
            if word in self.transitions[1]:
                if next_word in self.transitions[1][word]:
                    self.transitions[1][word][next_word] += 1
                else:
                    self.transitions[1][word][next_word] = 1
            else:
                self.transitions[1][word] = {}
                self.transitions[1][word][next_word] = 1

        for k_1,tdict in self.transitions[1].items():
            key_sum = float(sum(self.transitions[1][k_1].values()))
            for k_2,v in tdict.items():
                self.transitions[1][k_1][k_2] = v/key_sum 

    def generate_phrase(self):
        """
        Take a random sample from the probability distribution.  Terminate
        when a period, question mark, or exclamation point is drawn.
        """

        w_minus_1 = '?'
        while w_minus_1 in self.terminal_characters:
            w_minus_1 = np.random.choice(self.transitions[0].keys(),replace=True,p=self.transitions[0].values())
        phrase = w_minus_1+' '
        while w_minus_1 not in self.terminal_characters:
            w_minus_1 = np.random.choice(self.transitions[1][w_minus_1].keys(),replace=True,p=self.transitions[1][w_minus_1].values())
            phrase += w_minus_1+' '
        return phrase

    def compute_log_likelihood(self,phrase,lamda=0.0,unknown_probability=1e-5):
        """
        Return the log-probability of a given phrase (entered as a string)
        lambda: regularization factor for unseen transitions
        unknown_probability: probability mass of a word not in the dictionary.
        """

        words_in = phrase.split()
        
        w_i = words_in[0]
        try: 
            log_prob = np.log(self.transitions[0][w_i])
        except KeyError:
            log_prob = np.log(unknown_probability)
        for w in words_in[1:]:
            try:
                fjk = 0
                if w in self.transitions[1][w_i]:
                    fjk = self.transitions[1][w_i][w]
                log_prob += np.log((1-lamda)*fjk + lamda*self.transitions[0][w])
                w_i = w
            except KeyError:
                log_prob += np.log(unknown_probability)
        return log_prob



class SecondOrderMarkovModel(object):
    """
    Implements a trigram model for text generation

    Attributes:
        terminal_characters: a set of punctuation that shouldn't fall at the
                             beginning of a sentence, but should end one.

    Methods:
        build_transition_matrices: builds a dictionary of word frequency
        generate_phrase: generates a phrase by randomly sampling from 
                         word frequencies

    """


    terminal_characters = ['.','?','!']

    def __init__(self,sequence):
        """
        sequence: an ordered list of words corresponding to the training set
        """

        self.order=2
        self.sequence = sequence
        self.sequence_length = len(sequence)
        self.transitions = [{}]
        for i in range(self.order):
            self.transitions.append({})

    def build_transition_matrices(self):
        """
        Builds a set of nested dictionaries of word probabilities
        """

        for i in range(self.sequence_length):
            word = self.sequence[i]
            if word in self.transitions[0]:
                self.transitions[0][word] += 1
            else:
                self.transitions[0][word] = 1

        transition_sum = float(sum(self.transitions[0].values()))
        for k,v in self.transitions[0].items():
            self.transitions[0][k] = v/transition_sum

        for i in range(self.sequence_length-1):
            word = self.sequence[i]
            next_word = self.sequence[i+1]
            if word in self.transitions[1]:
                if next_word in self.transitions[1][word]:
                    self.transitions[1][word][next_word] += 1
                else:
                    self.transitions[1][word][next_word] = 1
            else:
                self.transitions[1][word] = {}
                self.transitions[1][word][next_word] = 1

        for k_1,tdict in self.transitions[1].items():
            key_sum = float(sum(self.transitions[1][k_1].values()))
            for k_2,v in tdict.items():
                self.transitions[1][k_1][k_2] = v/key_sum 

        for i in range(self.sequence_length-2):
            word = self.sequence[i]
            next_word = self.sequence[i+1]
            next_next_word = self.sequence[i+2]
            if word in self.transitions[2]:
                if next_word in self.transitions[2][word]:
                    if next_next_word in self.transitions[2][word][next_word]:
                        self.transitions[2][word][next_word][next_next_word] += 1
                    else:
                        self.transitions[2][word][next_word][next_next_word] = 1
                else:
                    self.transitions[2][word][next_word] = {}
                    self.transitions[2][word][next_word][next_next_word] = 1
            else:
                self.transitions[2][word] = {}
                self.transitions[2][word][next_word] = {}
                self.transitions[2][word][next_word][next_next_word] = 1
                    
        for k_1,tdict_1 in self.transitions[2].items():
            for k_2,tdict_2 in tdict_1.items():
                key_sum = float(sum(tdict_2.values()))
                for k_3,v in tdict_2.items():
                    self.transitions[2][k_1][k_2][k_3] = v/key_sum

    def generate_phrase(self):
        """
        Take a random sample from the probability distribution.  Terminate
        when a period, question mark, or exclamation point is drawn.
        """
        w_minus_2 = '?'
        while w_minus_2 in self.terminal_characters:
            w_minus_2 = np.random.choice(self.transitions[0].keys(),replace=True,p=self.transitions[0].values())
        w_minus_1 = np.random.choice(self.transitions[1][w_minus_2].keys(),replace=True,p=self.transitions[1][w_minus_2].values())
        phrase = w_minus_2 + ' ' + w_minus_1 + ' '
        while w_minus_1 not in self.terminal_characters:
            t_mat = self.transitions[2][w_minus_2][w_minus_1]
            w_i = np.random.choice(t_mat.keys(),replace=True,p=t_mat.values())
            phrase += w_i + ' '
            w_minus_2 = w_minus_1
            w_minus_1 = w_i
        return phrase


