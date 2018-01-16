# -*- coding: utf-8 -*-
# import modules & set up logging
import PreProcessor
import numpy as np

def main():
    print("Programm is starting...")
   # for code in PreProcessor.code_to_list():
	#print(code.cleared_code)
    for paper in PreProcessor.csvimport():
	for line in paper.original_paper:
		print (line)
    

if __name__ == "__main__":
    main()