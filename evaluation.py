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
codes = PreProcessor.code_to_list()

vocabulary = {}
vocabulary = set()

#Build vocabulary from codes und papers
for code in codes:
    vocabulary.update(code.cleared_code.split())

numberOfSentences = 0

for paper in papers:
    numberOfSentences += len(paper.cleared_paper)
    for line in paper.cleared_paper:
        vocabulary.update(line.split())

vocabulary = list(vocabulary)

print("Codes " + str(len(codes)))
print("sents " + str(numberOfSentences))

numberOfScores = len(codes) * numberOfSentences

score = np.zeros(shape=(numberOfScores,4))

print(score.shape)

infersentEncoder = InferSent(vocabulary=vocabulary)

# Get code embeddings
for code in codes:
    code.embedding = infersentEncoder.get_sent_embeddings([code.cleared_code])

index = 0

for paper in papers:
    print("Encoding paper id " + str(paper.id))
    paper_embeddings = infersentEncoder.get_sent_embeddings(paper.cleared_paper)
    for sentence_id in range(len(paper.cleared_paper)):
        for code in codes:
            score[index, 0] = paper.id
            score[index, 1] = sentence_id
            score[index, 2] = code.id
            score[index, 3] = infersentEncoder.cosine(code.embedding,paper_embeddings[sentence_id])
            index += index

np.savetxt("score.csv", score, delimiter=";")
#csv = np.genfromtxt('score.csv', delimiter=";")


#print(vocabulary)
# initialize InferSent Encoder with vocabulary
#infersentEncoder = InferSent(vocabulary=vocabulary)
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

