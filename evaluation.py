
import PreProcessor
import numpy as np
import os



print(os.getcwd())

for paper in PreProcessor.csvimport():
    print("title: " + paper.title)
    print(paper.cleared_paper)