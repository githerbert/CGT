# -*- coding: utf-8 -*-
# import modules & set up logging
import PreProcessor
import numpy as np
from Definitions import ROOT_DIR
import os, PreProcessor
from InferSent.encoder.infersent import InferSent
from code import Code
import sent2vec.sent2vec_encoder

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

#infersentEncoder = InferSent(vocabulary=vocabulary)


def get_infersent_score():
    infersentEncoder = InferSent()

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
                print(paper.cleared_paper[int(score[index,1])] + " // " + codes[int(score[index,2])].cleared_code + " //// " + str(score[index, 3]))
                print(score[index,:])
                index += 1

    print(infersentEncoder.cosine(infersentEncoder.model.encode(['our finding suggest that the analysis of mouse cursor movement may enable researcher to assess negative emotional reaction during live system use examine emotional reaction with more temporal precision conduct multimethod emotion research and provide researcher and system designer with an easy to deploy but powerful tool to infer user negative emotion to create more unobtrusive affective and adaptive system'])[0], infersentEncoder.model.encode(['the value of automatically analyze external datum'])[0]))

    print(score)

    os.chdir(ROOT_DIR)
    np.savetxt("score.csv", score, delimiter=";", fmt='%1.3f')
    #csv = np.genfromtxt('score.csv', delimiter=";")

def get_sent2vec_score():

    # Get code embeddings
    codelist = []
    for code in codes:
        codelist.append(code.cleared_code)
    code_embeddings = sent2vec.sent2vec_encoder.get_sentence_embeddings(codelist, ngram='unigrams', model='toronto')
    for i in range(len(codes)):
        codes[i].embedding = code_embeddings[i]

    index = 0

    for paper in papers:
        print("Encoding paper id " + str(paper.id))
        paper_embeddings = sent2vec.sent2vec_encoder.get_sentence_embeddings(paper.cleared_paper, ngram='unigrams', model='toronto')
        for sentence_id in range(len(paper.cleared_paper)):
            for code in codes:
                score[index, 0] = paper.id
                score[index, 1] = sentence_id
                score[index, 2] = code.id
                score[index, 3] = sent2vec.sent2vec_encoder.cosine(code.embedding, paper_embeddings[sentence_id])
                print(paper.cleared_paper[int(score[index, 1])] + " // " + codes[int(score[index, 2])].cleared_code + " //// " + str(score[index, 3]))
                index += 1

    sents = []
    sents.append('our finding suggest that the analysis of mouse cursor movement may enable researcher to assess negative emotional reaction during live system use examine emotional reaction with more temporal precision conduct multimethod emotion research and provide researcher and system designer with an easy to deploy but powerful tool to infer user negative emotion to create more unobtrusive affective and adaptive system')
    sents.append('the value of automatically analyze external datum')
    embeddings = sent2vec.sent2vec_encoder.get_sentence_embeddings(sents, ngram='unigrams', model='toronto')
    print(sent2vec.sent2vec_encoder.cosine(embeddings[0],embeddings[1]))

#get_sent2vec_score()
get_infersent_score()

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

