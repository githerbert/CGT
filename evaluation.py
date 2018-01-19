# -*- coding: utf-8 -*-
# import modules & set up logging
import PreProcessor
import numpy as np
from Definitions import ROOT_DIR
import os

os.chdir(ROOT_DIR)
print(os.getcwd())

for paper in PreProcessor.csvimport():
    print("title:  "+ paper.title)
    print(paper.cleared_paper)