# -*- coding: utf-8 -*-
# import modules & set up logging
import PreProcessor
import numpy as np

def main():
    print("Programm is starting...")
   # for code in PreProcessor.code_to_list():
	#print(code.cleared_code)
    score = np.zeros(shape=(50,3))
    score[25,0] = 150
    print(score)
    print(score.max(axis=0)[0])
    score = np.zeros(shape=(10,3))
    score[0,0] = 1
    score[1,0] = 2
    score[2,0] = 3
    score[3,0] = 4
    score[4,0] = 5
    print(np.percentile(score, 50, axis=0)[0])
    PreProcessor.csvexport()
    

if __name__ == "__main__":
    main()