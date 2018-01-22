# -*- coding: utf-8 -*-
# import modules & set up logging
import os, io,re, nltk
from nltk.tokenize import sent_tokenize
from Definitions import ROOT_DIR, PAPER_DIR, CODES_PATH, OS_NAME, LEM, STOP
import fnmatch
from preprocessing_lib import CONTRACTIONS_DICT, ABBREVATIONS_DICT, DASHES_LIST
import time
import spacy
from paper import Paper
from code import Code
import unicodecsv

# iterate through all subdirectories recusively and store all Main texts to a list
def iterate_folder(root):

    rootdir = root
    i = 0

    mainfiles = []

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if fnmatch.fnmatch(file, 'Main.txt'):
                mainfiles.append(os.path.join(subdir, file))
    return mainfiles

# Write text from a given "cp1252" encoded file in a utf-8 file, read it and store it in a list
def convertEncoding(filename, encoding='utf-8'):

    list = []
    if OS_NAME == "Windows":
        writefile = io.open(ROOT_DIR+'//ConvertEncoding.txt', 'r+', encoding="utf-8-sig")
    else:
        writefile = io.open(ROOT_DIR + '/ConvertEncoding.txt', 'r+', encoding="utf-8-sig")
    writefile.truncate(0)
    readfile = io.open(filename, encoding="cp1252")
    for line in readfile:
        writefile.write(line)
        writefile.seek(0,0)
        list.append(writefile.read())
        writefile.truncate(0)
        writefile.seek(0, 0)

    readfile.close()
    writefile.close()

    return list

# Extract codes from a file and store them in a list
def read_codes(filename):

    readfile = io.open(filename, encoding='utf16')

    cleared_code_list = []

    for line in readfile:

        words = line.split()

        if not all(word[0].isupper() for word in words) and words[-1] != ".":

            sentence = ' '.join([word for word in words])
            cleared_code_list.append(sentence)

    # Original codes
    original_codes = cleared_code_list[:]

    for i in range(len(cleared_code_list)):

        # Remove text between brackets
        cleared_code_list[i] = re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", cleared_code_list[i])

        # Expand contractions
        cleared_code_list[i] = expand_contractions(cleared_code_list[i])

        # Expand abbrevations
        cleared_code_list[i] = expand_abbrevations(cleared_code_list[i])

        # lowercasing
        cleared_code_list[i] = cleared_code_list[i].lower()

        tagged_words = []

        doc = nlp(cleared_code_list[i])

        for item in doc:
            if (item == doc[0] and "VB" in item.tag_) or (STOP == True and item.is_stop == True):
                pass
            # Splitt dash compounded words and singularize them
            elif "-" in item.text:
                dash_seperated_words = nlp(re.sub(r'([^a-zA-Z_]|_)+', ' ', item.text))
                for dash_word in dash_seperated_words:
                        word = dash_word.text
                        tag = dash_word.tag_
                        if tag == "NNS" and LEM == True:
                            word = item.lemma_
                        tagged_words.append(word)
            #else if not item.is_stop
            else:
                word = item.text
                tag = item.tag_
                if tag == "NNS" and LEM == True:
                    word = item.lemma_
                if "VB" in tag and LEM == True:
                    word = item.lemma_
                tagged_words.append(word)

        for j in range(len(tagged_words)):
            # Remove punctations
            tagged_words[j] = re.sub(r'([^a-zA-Z_]|_)+', '', tagged_words[j])

        #Remove redundant whitespaces
        cleared_code_list[i] = ' '.join(tagged_words)

        cleared_words = cleared_code_list[i].split()

        cleared_code_list[i] = ' '.join(cleared_words)

        # Replace semicolon in original codes
        original_codes[i] = original_codes[i].replace(";", ",")

    final_codes = []
    final_codes.append(original_codes)
    final_codes.append(cleared_code_list)

    return final_codes

# Extract text from file and store its sentences in a list
def sent_tokenize_file(filename):

    list = convertEncoding(filename)
    readfile = io.open(filename.replace("Main.txt", "Title.txt"), encoding="cp1252")
    title = ' '.join([line for line in readfile])
    title = title.replace(";", ",")

    fulltext = ""

    for line in list:
        words = line.split()
        #Remove punctutation for heading check
        normalizedHeading = []
        for i in range(len(words)):
            normalizedWord = re.sub(r'([^a-zA-Z_]|_)+', '', words[i])
            if len(normalizedWord) > 0:
                normalizedHeading.append(normalizedWord)

        if len(normalizedHeading) > 0:
            # Remove Headings
            if not all(word[0].isupper() for word in normalizedHeading) and words[-1] != ".":
                sentence = ' '.join([word for word in words])
                fulltext = ' '.join((fulltext, sentence))

    # Remove et al. from corpus
    fulltext = fulltext.replace(" et al.", "")
    # Normalize Dashes
    fulltext = normalize_dashes(fulltext)

    # Expand abbrevations
    fulltext = expand_abbrevations(fulltext)
    # Tokenize sentences
    sent_tokenize_list = sent_tokenize(fulltext)
    sent_tokenize_list_copy = sent_tokenize_list[:]

    cleared_list = []
    original_list = []

    for i in range(len(sent_tokenize_list)):

         words = sent_tokenize_list[i].split()

         # Remove sentences that contain more than 150 words
         if len(words) > 150:
             temp_list = []
             # Check if sentence is longer than 150 words with all invalid words removed
             for j in range(len(words)):
                if words[j] in nlp.vocab:
                    temp_list.append(words[j])
             if len(temp_list) > 150:
                 sent_tokenize_list[i] = ""
             else:
                 sent_tokenize_list[i] = ' '.join(temp_list)

         # Expand contractions
         sent_tokenize_list[i] = expand_contractions(sent_tokenize_list[i])

         # lowercasing
         sent_tokenize_list[i] = sent_tokenize_list[i].lower()

         words = sent_tokenize_list[i].split()

         for j in range(len(words)):

             # Join dash-seperated words if valid words are seperated
             if len(words[j]) > 0 and words[j][-1] == ("-") and j+1 < len(words):
                 if (re.sub(r'([^a-zA-Z_]|_)+', '', words[j]) + re.sub(r'([^a-zA-Z_]|_)+', '', words[j+1])) in nlp.vocab:
                     words[j] = words[j].replace("-", "") + words[j+1]
                     words[j+1] = ""

         sent_tokenize_list[i] = ' '.join(words)

         # Replace slashes with whitespace
         sent_tokenize_list[i] = sent_tokenize_list[i].replace("/", " ")

         sent_tokenize_list[i] = sent_tokenize_list[i].replace(u"\u2019", "")

         norm_word_list = []

         doc = nlp(sent_tokenize_list[i])

         for item in doc:
            if STOP == True and item.is_stop == True:
                pass
            # Split dash compounded words and singularize them
            elif "-" in item.text:
                dash_seperated_words = nlp(re.sub(r'([^a-zA-Z_]|_)+', ' ', item.text))
                for dash_word in dash_seperated_words:
                    word = dash_word.text
                    tag = dash_word.tag_
                    if tag == "NNS":
                        word = item.lemma_
                    norm_word_list.append(word)
            #else if not item.is_stop
            else:
            # Split dash compounded words and singularize them
                word = item.text
                tag = item.tag_
                if tag == "NNS" and LEM == True:
                    word = item.lemma_
                if "VB" in tag and LEM == True:
                    word = item.lemma_
                norm_word_list.append(word)

         sent_tokenize_list[i] = ' '.join(norm_word_list)

         words = sent_tokenize_list[i].split()

         for j in range(len(words)):
             # Remove Stop Words

             # Remove punctations and numbers
             words[j] = re.sub(r'([^a-zA-Z_]|_)+', '', words[j])
         sent_tokenize_list[i] = ' '.join(words)

         words = sent_tokenize_list[i].split()
         validWords = False
         letters = False
         for word in words:
             # Remove sententeces that contain no valid words
             if word in nlp.vocab:
                 validWords = True
             # Remove sententeces that contain letters only
             if len(word) > 1:
                 letters = True
         # Remove sentences that contain less than one word
         if validWords == True and letters == True and len(words)>1:
             cleared_list.append(' '.join(words))
             original_list.append(sent_tokenize_list_copy[i].replace(";", ","))
 
    final_list = []
    final_list.append(original_list)
    final_list.append(cleared_list)
    final_list.append(title)
    final_list.append(filename)
    print("--- %s seconds ---" % (time.time() - start_time))
    return final_list


def expand_contractions(s, contractions_dict=CONTRACTIONS_DICT):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

def expand_abbrevations(s, abbrevations_dict=ABBREVATIONS_DICT):

    for key in abbrevations_dict:
        s = s.replace(key,abbrevations_dict[key])
    return s

def normalize_dashes(s):
    for dash in DASHES_LIST:
        s = s.replace(dash,"-")
    return s

start_time = time.time()
nlp = spacy.load('en', disable=['parser', 'ner', 'textcat'])
print(nlp.pipeline)
contractions_re = re.compile('(%s)' % '|'.join(CONTRACTIONS_DICT.keys()))

def code_to_list():
    codes = []
    codelist = read_codes(CODES_PATH)
    for i in range(len(codelist[0])):
        code = Code(i,codelist[0][i],codelist[1][i])
        codes.append(code)
    return codes

def csvexport():

    #store paths of all main-texts in the given directory to a list
    filelist = iterate_folder(PAPER_DIR)
    numberOfPapers = len(filelist)


    # number of the paper
    i = 0
        #for line in sent_list[0]:
         #   print(line)
    with open('./export/sentences.csv','w') as s:
        with open('./export/papers.csv', 'w') as p:
            s.write("Paper_ID;Original;PreProcessed")
            s.write('\n')
            p.write("Paper_ID;Title;Path")
            p.write('\n')
            # iterate through all main texts and print their sentences
            for file in filelist:
                print("Paper with the ID " + str(i)+ " is currently written to csv-file... " + str(i)+ " / " + str(numberOfPapers))
                sent_list = sent_tokenize_file(file)
                p.write(str(i).encode('utf-8-sig') + ';'+ sent_list[2].encode('utf-8-sig') + ';'+ sent_list[3].encode('utf-8-sig'))
                p.write('\n')

                j = 0

                for item in sent_list[0]:
                    s.write(str(i).encode('utf-8-sig') + ';' + sent_list[0][j].encode('utf-8-sig') + ';'+ sent_list[1][j].encode('utf-8-sig'))
                    s.write('\n')
                    j = j + 1

                i += 1
        p.close()
    s.close()

def csvimport():

    preprocessed_paper_list = []

    myfile = open('./export/sentences.csv')
    sentences_data = unicodecsv.reader((x.replace('\0', '') for x in myfile), encoding='utf-8-sig', delimiter=';')
    sentences_data.next()

    papers = open('./export/papers.csv')
    papers_data = unicodecsv.reader((x.replace('\0', '') for x in papers), encoding='utf-8-sig', delimiter=';')
    papers_data.next()

    #Read Papers
    for row in papers_data:
        paper = Paper(int(row[0]), [], [], row[1], row[2])
        preprocessed_paper_list.append(paper)

    #Read sentences
    for row in sentences_data:
        preprocessed_paper_list[int(row[0])].original_paper.append(row[1])
        preprocessed_paper_list[int(row[0])].cleared_paper.append(row[2])

    return preprocessed_paper_list


def paper_to_list():

    filelist = iterate_folder(PAPER_DIR)

    # number of the paper
    i = 0

    preprocessed_paper_list = []

    # iterate through all main texts and print their sentences
    for file in filelist:
        
        print(i)
        sent_list = sent_tokenize_file(file)
        paper = Paper(i,sent_list[0],sent_list[1], sent_list[2], sent_list[3])
        preprocessed_paper_list.append(paper)
    i += 1

    return preprocessed_paper_list

#csvimport()
#csvexport()
#print(paper_to_list()[0].cleared_paper)

# //store the sentences in a file
# writefile = io.open('S:\\VMs\\Shared\\Maindata.txt', 'w', encoding="utf-8-sig")
# for file in filelist:
#     i+=1
#     print(i)
#     for line in tokenize_file(file):
#         writefile.write(line + "\n")
#
# writefile.close()