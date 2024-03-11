from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from xml.etree import ElementTree as ET
from collections import Counter


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    #  https://stackoverflow.com/questions/17140886/how-to-search-and-replace-text-in-a-file
    # Read in the file
    with open(filename, 'r') as file:
        filedata = file.read()
    # Replace the target string
    filedata = filedata.replace('&', '&amp;')
    # Write the file out again
    with open(filename, 'w') as file:
        file.write(filedata)

    tree = ET.parse(filename)
    root = tree.getroot()

    sentence_pairs = []
    alignments = []
    
    for child in root:
        en_sentence = child.find("english").text.split()
        cz_sentence = child.find("czech").text.split()
        sentence_pairs.append(SentencePair(en_sentence, cz_sentence))
        
        sure_pairs = child.find("sure").text

        if sure_pairs is not None:
            sure_pairs = [tuple(map(int, pair.split("-"))) for pair in sure_pairs.split()]
        else:
            sure_pairs = []

        possible_pairs = child.find("possible").text

        if possible_pairs is not None:
            possible_pairs = [tuple(map(int, pair.split("-"))) for pair in possible_pairs.split()]
        else:
            possible_pairs = []

        alignments.append(LabeledAlignment(sure_pairs, possible_pairs))
    
    return sentence_pairs, alignments

def get_token_to_index_for_language(word_counter, freq_cutoff):
    word_to_index = {}

    if freq_cutoff is None:
        word_to_index = {word: index for index, word in enumerate(word_counter)}
    else:
        word_to_index = {word[0]: index for index, word in enumerate(word_counter.most_common(freq_cutoff))}

    return word_to_index


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    source_counter = Counter()
    target_counter = Counter()

    for sentence_pair in sentence_pairs:
        for src_word in sentence_pair.source:
            source_counter[src_word] += 1
        for target_word in sentence_pair.target:
            target_counter[target_word] += 1
    
    source_dict = get_token_to_index_for_language(source_counter, freq_cutoff)
    target_dict = get_token_to_index_for_language(target_counter, freq_cutoff)
    
    return source_dict, target_dict

def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_sentence_pairs = []

    for sentence_pair in sentence_pairs:
        src_tokenized = []
        target_tokenized = []

        for src_word in sentence_pair.source:
            if src_word in source_dict:
                src_index = source_dict[src_word]
                src_tokenized.append(src_index)

        for target_word in sentence_pair.target:
            if target_word in target_dict:
                target_index = target_dict[target_word]
                target_tokenized.append(target_index)
        
        if len(src_tokenized) > 0 and len(target_tokenized) > 0:
            tokenized_sentence_pairs.append(TokenizedSentencePair(np.array(src_tokenized, dtype=np.int32), np.array(target_tokenized, dtype=np.int32)))

    return tokenized_sentence_pairs    
