# -*- coding: utf-8 -*-
# import modules & set up logging
import PreProcessor
import processor
from Definitions import SENT2VEC, INFERSENT

def main():
    print("Programm is starting...")
    # print("Pre-processing papers... (Estimated time: 1 hour)")
    # PreProcessor.csvexport()
    # print("Loading pre-processed codes and papers...")
    processor.load_codes_and_papers()
    #
    # print("Encoding papers... (Estimated time: 1 hour)")
    # if SENT2VEC == True and INFERSENT == False:
    #     processor.get_sent2vec_score()
    # elif INFERSENT == True and SENT2VEC == False:
    #     processor.get_infersent_score()

    print("Printing paper ranking... ")
    if SENT2VEC == True and INFERSENT == False:
        processor.print_sent2vec_relevance_ranking()
    elif INFERSENT == True and SENT2VEC == False:
        processor.print_infersent_relevance_ranking()
    

if __name__ == "__main__":
    main()