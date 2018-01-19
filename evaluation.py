# -*- coding: utf-8 -*-
# import modules & set up logging
import PreProcessor
import numpy as np
from Definitions import ROOT_DIR
import os
from InferSent.encoder.infersent import InferSent

os.chdir(ROOT_DIR)

infersentEncoder = InferSent()

print(infersentEncoder.get_sent_embeddings("Hello"))

# for paper in PreProcessor.csvimport():
#     print("title:  "+ paper.title)
#     print(paper.cleared_paper)

