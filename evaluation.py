# -*- coding: utf-8 -*-
# import modules & set up logging
import PreProcessor
import numpy as np
from Definitions import ROOT_DIR, LEM, STOP
import os, PreProcessor
from InferSent.encoder.infersent import InferSent
from code import Code
import sent2vec.sent2vec_encoder
import time
import spacy

os.chdir(ROOT_DIR)

papers = PreProcessor.csvimport()
codes = PreProcessor.code_to_list()

numberOfPapers = len(papers)

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


#infersentEncoder = InferSent(vocabulary=vocabulary)


def get_infersent_score():

    score = np.zeros(shape=(numberOfScores, 4))

    print(score.shape)

    infersentEncoder = InferSent(vocabulary=vocabulary)

    # Get code embeddings
    codelist = []
    for code in codes:
        codelist.append(code.cleared_code)
    code_embeddings = infersentEncoder.get_sent_embeddings(codelist)
    for i in range(len(codes)):
        codes[i].embedding = code_embeddings[i]

    index = 0

    start_time = time.time()

    for paper in papers:
        print("Encoding paper " + str(paper.id) + " / "+ str(numberOfPapers))
        paper_embeddings = infersentEncoder.get_sent_embeddings(paper.cleared_paper)
        for sentence_id in range(len(paper.cleared_paper)):
            for code in codes:
                score[index, 0] = paper.id
                score[index, 1] = sentence_id
                score[index, 2] = code.id
                score[index, 3] = infersentEncoder.cosine(code.embedding,paper_embeddings[sentence_id])
                index += 1

        print("--- %s seconds ---" % (time.time() - start_time))

    os.chdir(ROOT_DIR)
    #np.savetxt("score.csv", score, delimiter=";", fmt='%1.3f')
    #csv = np.genfromtxt('score.csv', delimiter=";")
    np.save('infersent_score.npy', score)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def get_infersent_score2():

    score = np.zeros(shape=(numberOfScores, 4))

    print(score.shape)

    infersentEncoder = InferSent(vocabulary=vocabulary)

    # Get code embeddings
    codelist = []
    for code in codes:
        codelist.append(code.cleared_code)
    code_embeddings = infersentEncoder.get_sent_embeddings(codelist)
    for i in range(len(codes)):
        codes[i].embedding = code_embeddings[i]

    index = 0
    papercount = 0

    start_time = time.time()

    #infersentEncoder = None

    for paper in papers:

        print("Encoding paper " + str(paper.id) + " / "+ str(numberOfPapers))
        sent_embedding = np.zeros(shape=(0,4096))
        sent_chunks = chunks(paper.cleared_paper, 5)
        for chunk in sent_chunks:
            #print(chunk)
            chunk_embedding = infersentEncoder.get_sent_embeddings(chunk)
            #print(chunk_embedding.shape)
            sent_embedding = np.vstack((sent_embedding,chunk_embedding))
        #sent_embedding = infersentEncoder.get_sent_embeddings([paper.cleared_paper[sentence_id]])

        for sentence_id in range(len(paper.cleared_paper)):
            for code in codes:
                score[index, 0] = paper.id
                score[index, 1] = sentence_id
                score[index, 2] = code.id
                score[index, 3] = infersentEncoder.cosine(code.embedding,sent_embedding[sentence_id])
                index += 1

        print("--- %s seconds ---" % (time.time() - start_time))

    os.chdir(ROOT_DIR)
    #np.savetxt("score.csv", score, delimiter=";", fmt='%1.3f')
    #csv = np.genfromtxt('score.csv', delimiter=";")
    np.save('infersent_score.npy', score)

def get_sent2vec_score():

    score = np.zeros(shape=(numberOfScores, 4))

    print(score.shape)

    # Get code embeddings
    codelist = []
    for code in codes:
        codelist.append(code.cleared_code)
    #code_embeddings = sent2vec.sent2vec_encoder.get_sentence_embeddings(codelist, ngram='unigrams', model='toronto')
    code_embeddings = sent2vec.sent2vec_encoder.encode(codelist)
    for i in range(len(codes)):
        codes[i].embedding = code_embeddings[i]

    index = 0

    start_time = time.time()

    for paper in papers:
        print("Encoding paper id " + str(paper.id))
        #paper_embeddings = sent2vec.sent2vec_encoder.get_sentence_embeddings(paper.cleared_paper, ngram='unigrams', model='toronto')
        paper_embeddings = sent2vec.sent2vec_encoder.encode(paper.cleared_paper)
        for sentence_id in range(len(paper.cleared_paper)):
            for code in codes:
                score[index, 0] = paper.id
                score[index, 1] = sentence_id
                score[index, 2] = code.id
                score[index, 3] = sent2vec.sent2vec_encoder.cosine(code.embedding, paper_embeddings[sentence_id])
                #print(paper.cleared_paper[int(score[index, 1])] + " // " + codes[int(score[index, 2])].cleared_code + " //// " + str(score[index, 3]))
                index += 1
        print("--- %s seconds ---" % (time.time() - start_time))


    #sents = []
    #sents.append('our finding suggest that the analysis of mouse cursor movement may enable researcher to assess negative emotional reaction during live system use examine emotional reaction with more temporal precision conduct multimethod emotion research and provide researcher and system designer with an easy to deploy but powerful tool to infer user negative emotion to create more unobtrusive affective and adaptive system')
    #sents.append('the value of automatically analyze external datum')
    #embeddings = sent2vec.sent2vec_encoder.get_sentence_embeddings(sents, ngram='unigrams', model='toronto')
    #print(sent2vec.sent2vec_encoder.cosine(embeddings[0],embeddings[1]))

    os.chdir(ROOT_DIR)
    #np.savetxt("score.csv", score, delimiter=";", fmt='%1.3f')
    np.save('sent2vec_score.npy', score)

def get_infersent_evaluation_samples():
    score = np.load('infersent_score.npy')
    #score = np.load('sent2vec_score.npy')
    score = np.nan_to_num(score)

    # infersent samples: 20 steps
    # 0.7 - 0.9 = 0,2

    print("score_column")
    score_column = score[:,3]
    print(score_column.shape)

    sample_array = np.zeros(shape=(0,4))

    i = 0.7

    while i < 0.9:
        # take 25 samples
        #result = np.where((score_column >= i) & (score_column < (i + 0.01)))
        result = np.argwhere((score_column >= i) & (score_column < (i + 0.01))).flatten()
        random_sample = np.random.choice(result,25,replace=False)

        #print(score[random_sample, :])
        for index in random_sample:
            print(score[index, :])
            sample_array = np.vstack((sample_array, score[index, :]))

        i += 0.01

    return sample_array

def get_sent2vec_evaluation_samples():

    score = np.load('sent2vec_score.npy')
    score = np.nan_to_num(score)

    # infersent samples: 20 steps
    # 0.7 - 0.9 = 0,2

    print("score_column")
    score_column = score[:,3]
    print(score_column.shape)

    sample_array = np.zeros(shape=(0,4))

    # sent2vec samples: 64 steps
    # 0.22 - 0.86 = 0,64

    i = 0.22
    count = 0
    sample_size = 7
    #
    while i < 0.86:
    #     # take 8 samples. First 12 steps only take 7 samples
        result = np.argwhere((score_column >= i) & (score_column < (i + 0.01))).flatten()

        if count == 12:
            sample_size = 8

        random_sample = np.random.choice(result,sample_size,replace=False)

        #print(score[random_sample, :])
        for index in random_sample:
            print(score[index, :])
            sample_array = np.vstack((sample_array, score[index, :]))

        i += 0.01
        count += 1

    return sample_array

def join_sample_arrays(infersent_sample, sent2vec_sample):

    joined_array = np.vstack((infersent_sample, sent2vec_sample))

    np.save('samples.npy', joined_array)

def get_score_for_specific_sentences():

    sentence_array = np.load('samples.npy')

    final_array = np.hstack((sentence_array, np.zeros(shape=(1000, 1))))

    sents = []

    original_codes = []
    original_sentences = []

    nlp = spacy.load('en', disable=['parser', 'ner', 'textcat'])

    for i in range(np.size(final_array,axis=0)):

        cleared_sentence = papers[int(final_array[i,0])].cleared_paper[int(final_array[i,1])]

        original_sentences.append(papers[int(final_array[i,0])].original_paper[int(final_array[i,1])])

        if LEM == True or STOP == True:

            norm_word_list = []

            doc = nlp(cleared_sentence)

            for item in doc:
                if STOP == True and item.is_stop == True:
                    pass
                # else if not item.is_stop
                else:
                    # Split dash compounded words and singularize them
                    word = item.text
                    tag = item.tag_
                    if tag == "NNS" and LEM == True:
                        word = item.lemma_
                    if "VB" in tag and LEM == True:
                        word = item.lemma_
                    norm_word_list.append(word)

            cleared_sentence = ' '.join(norm_word_list)

        sents.append(cleared_sentence)

    ################################## InferSent ############################################

    infersentEncoder = InferSent(vocabulary=vocabulary)

    # Get code embeddings
    codelist = []
    for code in codes:
        codelist.append(code.cleared_code)
    code_embeddings = infersentEncoder.get_sent_embeddings(codelist)
    for i in range(len(codes)):
        codes[i].embedding = code_embeddings[i]

    paper_embeddings = infersentEncoder.get_sent_embeddings(sents)

    for i in range(np.size(final_array, axis=0)):
        final_array[i,3] = infersentEncoder.cosine(paper_embeddings[i],codes[int(final_array[i,2])].embedding)
        original_codes.append(codes[int(final_array[i,2])].original_code)

    ################################## Sent2Vec #############################################

    # Get code embeddings
    codelist = []
    for code in codes:
        codelist.append(code.cleared_code)
    # code_embeddings = sent2vec.sent2vec_encoder.get_sentence_embeddings(codelist, ngram='unigrams', model='toronto')

    code_embeddings = sent2vec.sent2vec_encoder.encode(codelist)

    for i in range(len(codes)):
        codes[i].embedding = code_embeddings[i]

    paper_embeddings = sent2vec.sent2vec_encoder.encode(sents)

    for i in range(np.size(final_array, axis=0)):
        final_array[i, 4] = sent2vec.sent2vec_encoder.cosine(paper_embeddings[i], codes[int(final_array[i, 2])].embedding)

    os.chdir(ROOT_DIR)

    if STOP == True and LEM == True:
        np.savetxt("evaluation_stop_lem.csv", final_array, delimiter=";", fmt='%i %i %i %1.6f %1.6f')
    elif STOP == False and LEM == True:
        np.savetxt("evaluation_lem.csv", final_array, delimiter=";", fmt='%i %i %i %1.6f %1.6f')
    elif STOP == True and LEM == False:
        np.savetxt("evaluation_stop.csv", final_array, delimiter=";", fmt='%i %i %i %1.6f %1.6f')
    elif STOP == False and LEM == False:
        np.savetxt("evaluation.csv", final_array, delimiter=";", fmt='%i %i %i %1.6f %1.6f')

    with open('sent_codes.csv','w') as s:
        for i in range(np.size(final_array, axis=0)):
            s.write(original_sentences[i].encode('utf-8-sig') + ';'+ original_codes[i].encode('utf-8-sig'))
            s.write('\n')
    s.close()

#join_sample_arrays(get_infersent_evaluation_samples(),get_sent2vec_evaluation_samples())
get_score_for_specific_sentences()
#get_infersent_evaluation_samples()
#get_sent2vec_score()
#get_infersent_score()

