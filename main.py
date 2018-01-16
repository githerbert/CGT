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
   
    

if __name__ == "__main__":
    main()