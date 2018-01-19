# -*- coding: utf-8 -*-
# import modules & set up logging
import PreProcessor
import numpy as np
from Definitions import ROOT_DIR
import os, PreProcessor
from InferSent.encoder.infersent import InferSent
from code import Code

os.chdir(ROOT_DIR)

papers = PreProcessor.csvimport()

vocabulary = {}
vocabulary = set()

for paper in papers:
    for line in paper.cleared_paper:
        vocabulary.update(line.split())

vocabulary = list(vocabulary)

print(vocabulary)

infersentEncoder = InferSent(vocabulary=vocabulary)
#
# sentences = []
# sentences.append("Hello my name is")
# sentences.append("two woman going to the supermarket")
# sentences.append("two students going to the school")
#
# code1 = Code(0,[],[])
# code2 = Code(0,[],[])
#
# embeddings = infersentEncoder.get_sent_embeddings(sentences)
#
# code1.embedding = np.array(embeddings[1])
# code2.embedding = np.array(embeddings[2])
# print(infersentEncoder.get_sent_embeddings(sentences))
# print(infersentEncoder.cosine(code1.embedding,code2.embedding))
# infersentEncoder.model.visualize("two woman going to the supermarket")
# infersentEncoder.model.visualize("two students going to the school")

#print(infersentEncoder.get_sent_embeddings(sentences).shape)

# for paper in PreProcessor.csvimport():
#     print("title:  "+ paper.title)
#     print(paper.cleared_paper)

