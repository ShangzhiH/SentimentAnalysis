# encoding=utf-8

"""
该文件提供一些处理数据的函数
"""

import tensorflow as tf
import numpy as np
import re


__all__ = ["line_num_count"]


def load_word2vec(logger, emb_path, id_to_word, word_dim, old_weights):
    """
    load pretrained embedding vectors
    :param emb_path: 
    :param id_to_word: 
    :param word_dim: 
    :param old_weights: 
    :return: 
    """
    new_weights = old_weights
    logger.info("Loading pretrained embeddings from {}...".format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(tf.gfile.GFile(emb_path, "r")):
        line = line.rstrip().decode("utf-8").split(" ")
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array([float(x) for x in line[1:]]).astype(np.float32)
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        logger.info("Warning: %i invalid lines in embedding file" % emb_invalid)
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    for i in range(n_words):
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[
                re.sub('\d', '0', word.lower())
            ]
            c_zeros += 1
    logger.info("Loaded %i pretrained embeddings." % len(pre_trained))
    logger.info("%i / %i (%.4f%%) words have been initialized with pretrained embeddings." % (c_found + c_lower + c_zeros, n_words, 100.0 * (c_found + c_lower + c_zeros) / n_words))
    logger.info('%i found directly, %i after lowercasing, '
          '%i after lowercasing + zero.' % (
              c_found, c_lower, c_zeros
          ))
    return new_weights


def full_to_half(s):
    """
    Convert full-width character to half-width one 
    """
    if type(s) == str:
        s = s.decode("utf-8")
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = unichr(num)
        n.append(char)
    return u''.join(n)


def replace_html(s):
    if type(s) == str:
        s = s.decode("utf-8")

    s = s.replace(u'&quot;', u'"')
    s = s.replace(u'&amp;', u'&')
    s = s.replace(u'&lt;', u'<')
    s = s.replace(u'&gt;', u'>')
    s = s.replace(u'&nbsp;', u' ')
    s = s.replace(u"&ldquo;", u"“")
    s = s.replace(u"&rdquo;", u"”")
    s = s.replace(u"&mdash;", u"")
    s = s.replace(u"\xa0", u" ")
    return s


def line_num_count(file_path):
    """
    count sentence num
    """
    num = 0
    for line in tf.gfile.GFile(file_path, "r"):
        if line.strip():
            num += 1
    return num
