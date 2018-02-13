# -*- coding: utf-8 -*-
# import modules & set up logging
import PreProcessor
import numpy as np
from Definitions import ROOT_DIR, infersent_threshold, sent2vec_threshold, paper_details, INFERSENT, SENT2VEC
import os, PreProcessor
from InferSent.encoder.infersent import InferSent
import sent2vec.sent2vec_encoder
import time

os.chdir(ROOT_DIR)

papers = []
codes = []

numberOfPapers = 0

vocabulary = {}
vocabulary = set()

numberOfSentences = 0

numberOfScores = 0

def load_codes_and_papers():
    global papers
    global codes
    global numberOfPapers
    global vocabulary
    global numberOfSentences
    global numberOfScores

    papers = PreProcessor.csvimport()
    codes = PreProcessor.code_to_list()

    numberOfPapers = len(papers)

    # Build vocabulary from codes und papers
    for code in codes:
        vocabulary.update(code.cleared_code.split())

    for paper in papers:
        numberOfSentences += len(paper.cleared_paper)
        for line in paper.cleared_paper:
            vocabulary.update(line.split())

    vocabulary = list(vocabulary)

    print("Codes " + str(len(codes)))
    print("sents " + str(numberOfSentences))

    numberOfScores = len(codes) * numberOfSentences


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
    score = np.nan_to_num(score)
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
    score = np.nan_to_num(score)
    np.save('sent2vec_score.npy', score)

def print_sent2vec_relevance_ranking():

    score = np.load('sent2vec_score.npy')

    # Keep all scores which are greater than the threshold
    idx = np.where(score[:,3] > sent2vec_threshold)

    scores_above_threshold = score[idx]

    print(np.size(scores_above_threshold, axis=0))

    # Array for interview paper ranking

    interview_scores = np.zeros(shape=(0, 2))

    # number of codes in the interview

    m = len(codes)

    # Iterature through all codes

    for code in codes:
        print("Code " + str(code.id) + ": " + code.original_code)
        paper_code_id = np.where(scores_above_threshold[:, 2] == float(code.id))
        paper_code_array = scores_above_threshold[paper_code_id]

        # Find all corresponding papers to this code
        unique_papers = np.unique(paper_code_array[:, 0])

        if len(unique_papers) < 1:
            print("No corresponding literature found")
        else:
            # Create array for the relevance scores of the papers
            relevancescore = np.zeros(shape=(len(unique_papers), 2))
            for index in range(len(unique_papers)):
                relevancescore[index, 0] = unique_papers[index]
                sentence_index = np.where(paper_code_array[:, 0] == unique_papers[index])
                sentence_array = paper_code_array[sentence_index]
                # Iterate over all relevant sentences of the paper
                for s in range(np.size(sentence_array, axis=0)):
                    relevancescore[index, 1] = relevancescore[index, 1] + sentence_array[s, 3]

            relevancescore_sorted = np.flip(relevancescore[relevancescore[:, 1].argsort()], 0)
            for p in range(np.size(relevancescore_sorted, axis=0)):
                corr_paper = papers[int(relevancescore_sorted[p, 0])]
                weighted_papers = np.array(relevancescore_sorted[p, :], copy=True)
                weighted_papers[1] = weighted_papers[1] / m
                interview_scores = np.vstack((interview_scores, weighted_papers))
                print(" # " + str(p + 1) + " Paper_title: " + corr_paper.title + "    Relevance Score: " + str(
                    relevancescore_sorted[p, 1]))
                if paper_details == True:
                    sentence_index = np.where(paper_code_array[:, 0] == relevancescore_sorted[p, 0])
                    sentence_array = paper_code_array[sentence_index]
                    # Iterate over all relevant sentences of the paper
                    for s in range(np.size(sentence_array, axis=0)):
                        corr_sentence = papers[int(relevancescore_sorted[p, 0])].original_paper[
                            int(sentence_array[s, 1])]
                        print("Sentence no: " + str(s + 1) + " Similarity: " + str(
                            sentence_array[s, 3]) + "  " + corr_sentence)

    print(
        "########### INTERVIEW PAPER RANKING ######################################################################################")

    if m < 1 and len(papers) < 1:
        print("No corresponding literature found for this interview")
    else:
        unique_papers = np.unique(interview_scores[:, 0])

        # Create array for the interview relevance scores of the papers
        relevancescore = np.zeros(shape=(len(unique_papers), 2))
        for index in range(len(unique_papers)):
            relevancescore[index, 0] = unique_papers[index]
            paper_index = np.where(interview_scores[:, 0] == unique_papers[index])
            paper_array = interview_scores[paper_index]
            # Iterate over all weighted sums of the paper
            for s in range(np.size(paper_array, axis=0)):
                relevancescore[index, 1] = relevancescore[index, 1] + paper_array[s, 1]

        iv_relevancescore_sorted = np.flip(relevancescore[relevancescore[:, 1].argsort()], 0)

        for p in range(np.size(iv_relevancescore_sorted, axis=0)):
            corr_paper = papers[int(iv_relevancescore_sorted[p, 0])]
            print(" # " + str(p + 1) + " Paper_title: " + corr_paper.title + "   Weighted Relevance Score: " + str(
                iv_relevancescore_sorted[p, 1]))


def sent2vec_relevance_ranking_to_disk():

    score = np.load('sent2vec_score.npy')

    # Keep all scores which are greater than the threshold
    idx = np.where(score[:,3] > sent2vec_threshold)

    scores_above_threshold = score[idx]

    # Iterature through all codes

    with open('sent2vec_relevance_ranking.csv','w') as r:

        for code in codes:
            print("Code " + str(code.id) + ": " + code.original_code)
            paper_code_id = np.where(scores_above_threshold[:,2] == float(code.id))
            paper_code_array = scores_above_threshold[paper_code_id]

            # Find all corresponding papers to this code
            unique_papers = np.unique(paper_code_array[:,0])

            if len(unique_papers) < 1:
                print("No corresponding literature found")
            else:
                # Create array for the relevance scores of the papers
                relevancescore = np.zeros(shape=(len(unique_papers), 2))
                for index in range(len(unique_papers)):
                    relevancescore[index,0] = unique_papers[index]
                    sentence_index = np.where(paper_code_array[:,0] == unique_papers[index])
                    sentence_array = paper_code_array[sentence_index]
                    # Iterate over all relevant sentences of the paper
                    for s in range(np.size(sentence_array, axis=0)):
                        relevancescore[index, 1] = relevancescore[index, 1] + sentence_array[s,3]

                relevancescore_sorted = np.flip(relevancescore[relevancescore[:,1].argsort()],0)
                for p in range(np.size(relevancescore_sorted, axis=0)):
                    corr_paper = papers[int(relevancescore_sorted[p,0])]
                    print(" # "+ str(p+1) + " Paper_title: " + corr_paper.title + "    Relevance Score: " + str(relevancescore_sorted[p,1]))
                    if paper_details == True:
                        sentence_index = np.where(paper_code_array[:, 0] == relevancescore_sorted[p,0])
                        sentence_array = paper_code_array[sentence_index]
                        # Iterate over all relevant sentences of the paper
                        for s in range(np.size(sentence_array, axis=0)):
                            corr_sentence = papers[int(relevancescore_sorted[p, 0])].original_paper[int(sentence_array[s,1])]
                            r.write(code.original_code.encode('utf-8-sig') + ';' + corr_sentence.encode('utf-8-sig') + ";" + str(sentence_array[s,3]).encode('utf-8-sig') + ";" + corr_paper.title.encode('utf-8-sig') + ";" + corr_paper.path.encode('utf-8-sig'))
                            r.write('\n')

    r.close()

def print_infersent_relevance_ranking():

    score = np.load('infersent_score.npy')

    # Keep all scores which are greater than the threshold
    idx = np.where(score[:,3] > infersent_threshold)

    scores_above_threshold = score[idx]

    print(np.size(scores_above_threshold, axis=0))

    # Array for interview paper ranking

    interview_scores = np.zeros(shape=(0, 2))

    # number of codes in the interview

    m = len(codes)

    # Iterature through all codes

    for code in codes:
        print("Code " + str(code.id) + ": " + code.original_code)
        paper_code_id = np.where(scores_above_threshold[:,2] == float(code.id))
        paper_code_array = scores_above_threshold[paper_code_id]

        # Find all corresponding papers to this code
        unique_papers = np.unique(paper_code_array[:,0])

        if len(unique_papers) < 1:
            print("No corresponding literature found")
        else:
            # Create array for the relevance scores of the papers
            relevancescore = np.zeros(shape=(len(unique_papers), 2))
            for index in range(len(unique_papers)):
                relevancescore[index,0] = unique_papers[index]
                sentence_index = np.where(paper_code_array[:,0] == unique_papers[index])
                sentence_array = paper_code_array[sentence_index]
                # Iterate over all relevant sentences of the paper
                for s in range(np.size(sentence_array, axis=0)):
                    relevancescore[index, 1] = relevancescore[index, 1] + sentence_array[s,3]

            relevancescore_sorted = np.flip(relevancescore[relevancescore[:,1].argsort()],0)
            for p in range(np.size(relevancescore_sorted, axis=0)):
                corr_paper = papers[int(relevancescore_sorted[p,0])]
                weighted_papers = np.array(relevancescore_sorted[p, :], copy=True)
                weighted_papers[1] = weighted_papers[1]/m
                interview_scores = np.vstack((interview_scores, weighted_papers))
                print(" # "+ str(p+1) + " Paper_title: " + corr_paper.title + "    Relevance Score: " + str(relevancescore_sorted[p,1]))
                if paper_details == True:
                    sentence_index = np.where(paper_code_array[:, 0] == relevancescore_sorted[p,0])
                    sentence_array = paper_code_array[sentence_index]
                    # Iterate over all relevant sentences of the paper
                    for s in range(np.size(sentence_array, axis=0)):
                        corr_sentence = papers[int(relevancescore_sorted[p, 0])].original_paper[int(sentence_array[s,1])]
                        print("Sentence no: " + str(s+1) + " Similarity: " + str(sentence_array[s,3]) + "  " + corr_sentence)

    print("########### INTERVIEW PAPER RANKING ######################################################################################")

    if m < 1 and len(papers) < 1:
        print("No corresponding literature found for this interview")
    else:
        unique_papers = np.unique(interview_scores[:, 0])

        # Create array for the interview relevance scores of the papers
        relevancescore = np.zeros(shape=(len(unique_papers), 2))
        for index in range(len(unique_papers)):
            relevancescore[index, 0] = unique_papers[index]
            paper_index = np.where(interview_scores[:, 0] == unique_papers[index])
            paper_array = interview_scores[paper_index]
            # Iterate over all weighted sums of the paper
            for s in range(np.size(paper_array, axis=0)):
                relevancescore[index, 1] = relevancescore[index, 1] + paper_array[s, 1]

        iv_relevancescore_sorted = np.flip(relevancescore[relevancescore[:, 1].argsort()], 0)

        for p in range(np.size(iv_relevancescore_sorted, axis=0)):
            corr_paper = papers[int(iv_relevancescore_sorted[p, 0])]
            print(" # " + str(p + 1) + " Paper_title: " + corr_paper.title + "   Weighted Relevance Score: " + str(
                iv_relevancescore_sorted[p, 1]))


def infersent_relevance_ranking_to_disk():

    score = np.load('infersent_score.npy')

    # Keep all scores which are greater than the threshold
    idx = np.where(score[:,3] > infersent_threshold)

    scores_above_threshold = score[idx]

    # Iterature through all codes

    with open('infersent_relevance_ranking.csv','w') as r:

        for code in codes:
            print("Code " + str(code.id) + ": " + code.original_code)
            paper_code_id = np.where(scores_above_threshold[:,2] == float(code.id))
            paper_code_array = scores_above_threshold[paper_code_id]

            # Find all corresponding papers to this code
            unique_papers = np.unique(paper_code_array[:,0])

            if len(unique_papers) < 1:
                print("No corresponding literature found")
            else:
                # Create array for the relevance scores of the papers
                relevancescore = np.zeros(shape=(len(unique_papers), 2))
                for index in range(len(unique_papers)):
                    relevancescore[index,0] = unique_papers[index]
                    sentence_index = np.where(paper_code_array[:,0] == unique_papers[index])
                    sentence_array = paper_code_array[sentence_index]
                    # Iterate over all relevant sentences of the paper
                    for s in range(np.size(sentence_array, axis=0)):
                        relevancescore[index, 1] = relevancescore[index, 1] + sentence_array[s,3]

                relevancescore_sorted = np.flip(relevancescore[relevancescore[:,1].argsort()],0)
                for p in range(np.size(relevancescore_sorted, axis=0)):
                    corr_paper = papers[int(relevancescore_sorted[p,0])]
                    print(" # "+ str(p+1) + " Paper_title: " + corr_paper.title + "    Relevance Score: " + str(relevancescore_sorted[p,1]))
                    if paper_details == True:
                        sentence_index = np.where(paper_code_array[:, 0] == relevancescore_sorted[p,0])
                        sentence_array = paper_code_array[sentence_index]
                        # Iterate over all relevant sentences of the paper
                        for s in range(np.size(sentence_array, axis=0)):
                            corr_sentence = papers[int(relevancescore_sorted[p, 0])].original_paper[int(sentence_array[s,1])]
                            r.write(code.original_code.encode('utf-8-sig') + ';' + corr_sentence.encode('utf-8-sig') + ";" + str(sentence_array[s,3]).encode('utf-8-sig') + ";" + corr_paper.title.encode('utf-8-sig') + ";" + corr_paper.path.encode('utf-8-sig'))
                            r.write('\n')

    r.close()

