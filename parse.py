# -*- coding: utf-8 -*-

from nltk import CFG
import nltk
from nltk.chunk.regexp import *
from nltk.parse.stanford import StanfordParser
# from graphviz import Digraph
# import re

# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('ieer')
# 语法分析

class DisjointSet:
    def __init__(self, n, hi=False):
        self.fa = [i for i in range(n)]
        self.hi = []
        if hi:
            self.hi = [1 for _ in range(n)]

    def union(self, val1, val2):
        """
        normal union, make val1's father equal val2's father
        :param val1: value of node which need to change root
        :param val2: another node's value
        :return: None
        """
        self.fa[self.find(val1)] = self.find(val2)
        return None

    def find(self, val):
        """
        normal find, get father of val
        :param val: a node's value
        :return: root of val
        """
        if self.fa[val] == val:
            return val
        else:
            return self.find(self.fa[val])

    def union_compress(self, val1, val2):
        """
        union according to height of tree
        :param val1:  a node's value
        :param val2:  another node's value
        :return: None
        """
        x, y = self.find(val1), self.find(val2)
        if self.hi[x] <= self.hi[y]:
            self.fa[x] = y
        else:
            self.fa[y] = x
        if self.hi[x] == self.hi[y] and x != y:
            self.hi[y] += 1
        return None

    def find_compress(self, val):
        """
        find with path compress
        :param val:  a node's value
        :return: father of val, val 's father equal val's root
        """
        if self.fa[val] == val:
            return val
        else:
            self.fa[val] = self.find(self.fa[val])
            return self.fa[val]

    def show(self):
        return (self.fa)

# from stat_parser import Parser
# parser = Parser()
# print parser.parse("How can the net amount of entropy of the universe be massively decreased?")
words=['wow']
pos_tags = nltk.pos_tag(words)[0][1]
print(pos_tags)