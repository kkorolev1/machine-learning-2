from abc import ABC, abstractmethod
from itertools import product
from typing import List, Tuple

import numpy as np

from preprocessing import TokenizedSentencePair


class BaseAligner(ABC):
    """
    Describes a public interface for word alignment models.
    """

    @abstractmethod
    def fit(self, parallel_corpus: List[TokenizedSentencePair]):
        """
        Estimate alignment model parameters from a collection of parallel sentences.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
        """
        pass

    @abstractmethod
    def align(self, sentences: List[TokenizedSentencePair]) -> List[List[Tuple[int, int]]]:
        """
        Given a list of tokenized sentences, predict alignments of source and target words.

        Args:
            sentences: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            alignments: list of alignments for each sentence pair, i.e. lists of tuples (source_pos, target_pos).
            Alignment positions in sentences start from 1.
        """
        pass


class DiceAligner(BaseAligner):
    def __init__(self, num_source_words: int, num_target_words: int, threshold=0.5):
        self.cooc = np.zeros((num_source_words, num_target_words), dtype=np.uint32)
        self.dice_scores = None
        self.threshold = threshold

    def fit(self, parallel_corpus):
        for sentence in parallel_corpus:
            # use np.unique, because for a pair of words we add 1 only once for each sentence
            for source_token in np.unique(sentence.source_tokens):
                for target_token in np.unique(sentence.target_tokens):
                    self.cooc[source_token, target_token] += 1
        self.dice_scores = (2 * self.cooc.astype(np.float32) /
                            (self.cooc.sum(0, keepdims=True) + self.cooc.sum(1, keepdims=True)))

    def align(self, sentences):
        result = []
        for sentence in sentences:
            alignment = []
            for (i, source_token), (j, target_token) in product(
                    enumerate(sentence.source_tokens, 1),
                    enumerate(sentence.target_tokens, 1)):
                if self.dice_scores[source_token, target_token] > self.threshold:
                    alignment.append((i, j))
            result.append(alignment)
        return result


class WordAligner(BaseAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        self.num_source_words = num_source_words
        self.num_target_words = num_target_words
        self.translation_probs = np.full((num_source_words, num_target_words), 1 / num_target_words, dtype=np.float32)
        self.num_iters = num_iters

    def _e_step(self, parallel_corpus: List[TokenizedSentencePair]) -> List[np.array]:
        """
        Given a parallel corpus and current model parameters, get a posterior distribution over alignments for each
        sentence pair.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            posteriors: list of np.arrays with shape (src_len, target_len). posteriors[i][j][k] gives a posterior
            probability of target token k to be aligned to source token j in a sentence i.
        """
        posteriors = []

        for sentence in parallel_corpus:
            theta = self.translation_probs[np.ix_(sentence.source_tokens, sentence.target_tokens)]
            col_sum = theta.sum(axis=0) # shape of num_targets
            posteriors.append(theta / col_sum)
        
        return posteriors

    def _compute_elbo(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]) -> float:
        """
        Compute evidence (incomplete likelihood) lower bound for a model given data and the posterior distribution
        over latent variables.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo: the value of evidence lower bound
        """
        res = 0
        eps = 1e-9
        for i, sentence in enumerate(parallel_corpus):
            res += (posteriors[i] * (-np.log(posteriors[i] + eps) + np.log(eps + self.translation_probs[np.ix_(sentence.source_tokens, sentence.target_tokens)]))).sum()
            res -= len(sentence.target_tokens) * np.log(len(sentence.source_tokens))
        return res
    
    def _m_step(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]):
        """
        Update model parameters from a parallel corpus and posterior alignment distribution. Also, compute and return
        evidence lower bound after updating the parameters for logging purposes.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo:  the value of evidence lower bound after applying parameter updates
        """
        del self.translation_probs
        self.translation_probs = np.zeros((self.num_source_words, self.num_target_words), dtype=np.float32)
        for corpus, q in zip(parallel_corpus, posteriors):
            np.add.at(self.translation_probs, np.ix_(corpus.source_tokens, corpus.target_tokens), q)
        self.translation_probs /= self.translation_probs.sum(axis=1).reshape(-1, 1)
        return self._compute_elbo(parallel_corpus, posteriors)

    def fit(self, parallel_corpus):
        """
        Same as in the base class, but keep track of ELBO values to make sure that they are non-decreasing.
        Sorry for not sticking to my own interface ;)

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            history: values of ELBO after each EM-step
        """
        history = []
        for i in range(self.num_iters):
            posteriors = self._e_step(parallel_corpus)
            elbo = self._m_step(parallel_corpus, posteriors)
            history.append(elbo)
        return history

    def align(self, sentences):
            alignments = []
            for sentence in sentences:
                alignments.append([])
                for i in range(len(sentence.target_tokens)):
                    argmax_src, argmax_j = None, None
                    tgt = sentence.target_tokens[i]
                    for j in range(len(sentence.source_tokens)):
                        src = sentence.source_tokens[j]
                        if argmax_src is None or self.translation_probs[src][tgt] > self.translation_probs[argmax_src][tgt]:
                            argmax_src = src
                            argmax_j = j
                    alignments[-1].append((argmax_j + 1, i + 1))
            return alignments


class WordPositionAligner(WordAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        super().__init__(num_source_words, num_target_words, num_iters)
        self.alignment_probs = {}

    def _get_probs_for_lengths(self, src_length: int, tgt_length: int):
        """
        Given lengths of a source sentence and its translation, return the parameters of a "prior" distribution over
        alignment positions for these lengths. If these parameters are not initialized yet, first initialize
        them with a uniform distribution.

        Args:
            src_length: length of a source sentence
            tgt_length: length of a target sentence

        Returns:
            probs_for_lengths: np.array with shape (src_length, tgt_length)
        """
        if (src_length, tgt_length) not in self.alignment_probs:
            self.alignment_probs[(src_length, tgt_length)] = np.full((src_length, tgt_length), 1 / src_length, dtype=np.float32)
        return self.alignment_probs[(src_length, tgt_length)]

    def _e_step(self, parallel_corpus):
        posteriors = []

        for sentence in parallel_corpus:
            phis = self._get_probs_for_lengths(len(sentence.source_tokens), len(sentence.target_tokens))
            theta = self.translation_probs[np.ix_(sentence.source_tokens, sentence.target_tokens)]
            mat = phis * theta
            posteriors.append(mat / mat.sum(axis = 0))

        return posteriors

    def _compute_elbo(self, parallel_corpus, posteriors):
        res = np.float64(0)
        eps = 1e-9
        for sentence, q in zip(parallel_corpus, posteriors):
            phis = self._get_probs_for_lengths(len(sentence.source_tokens), len(sentence.target_tokens))
            mat = (np.log(eps + phis) +
                  np.log(eps + self.translation_probs[np.ix_(sentence.source_tokens, sentence.target_tokens)]) -
                  np.log(eps + q))
            res += (q * mat).sum()
        return res

    def _m_step(self, parallel_corpus, posteriors):
        del self.translation_probs
        self.translation_probs = np.zeros((self.num_source_words, self.num_target_words), dtype=np.float32)
        self.alignment_probs = {}

        for sentence, q in zip(parallel_corpus, posteriors):
            np.add.at(self.translation_probs, np.ix_(sentence.source_tokens, sentence.target_tokens), q)
            
            alignment_probs_key = (len(sentence.source_tokens), len(sentence.target_tokens))
            if alignment_probs_key not in self.alignment_probs:
                self.alignment_probs[alignment_probs_key] = q.copy()
            else:
                self.alignment_probs[alignment_probs_key] += q

        for key in self.alignment_probs:
            self.alignment_probs[key] /= self.alignment_probs[key].sum(axis = 0)

        self.translation_probs /= self.translation_probs.sum(axis=1).reshape(-1, 1)
        return self._compute_elbo(parallel_corpus, posteriors)
