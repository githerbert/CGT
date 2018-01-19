# import stuff

from random import randint
#import matplotlib

import numpy as np
import torch, os
from Definitions import ROOT_DIR

class InferSent:

    def __init__(self):
        self.GLOVE_PATH = ROOT_DIR + '/InferSent/dataset/GloVe/glove.840B.300d.txt'

        # LOAD MODEL

        # make sure models.py is in the working directory
        # model = torch.load('infersent.allnli.pickle')

        os.chdir(os.path.dirname(__file__))

        self.model = torch.load('infersent.allnli.pickle', map_location=lambda storage, loc: storage)

        torch.set_num_threads(2)

        # torch.load(..pickle) will use GPU/Cuda by default. If you are on CPU:
        # model = torch.load('infersent.allnli.pickle', map_location=lambda storage, loc: storage)
        # On CPU, setting the right number of threads with "torch.set_num_threads(k)" may improve performance

        self.model.set_glove_path(self.GLOVE_PATH)

        self.model.build_vocab_k_words(K=200000)

    #print(cosine(model.encode(['the cat eats.'])[0], model.encode(['the cat drinks.'])[0]))
    # print("The cosine similarity between the two sentences")
    # print('the cat eats the mice.'+' mice and cats.')
    # print("is:")
    # print(cosine(model.encode(['the cat eats the mice.'])[0], model.encode(['the mice are eaten by the cat.'])[0]))
    #
    # print(cosine(model.encode(['customers loyalty'])[0], model.encode(['customer loyality'])[0]))


    def get_sent_embeddings(self, sentences):
        return self.model.encode(sentences)


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
