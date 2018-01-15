# -*- coding: utf-8 -*-
# import modules & set up logging
import os, io,re, nltk
from nltk.tokenize import sent_tokenize
from Definitions import ROOT_DIR, PAPER_DIR, CODES_PATH, OS_NAME
from nltk.corpus import brown, stopwords
import fnmatch
from Contractions import CONTRACTIONS_DICT
from Abbrevations import ABBREVATIONS_DICT
from textblob import TextBlob, Word
from textblob.taggers import NLTKTagger
import time
import spacy
import csv
from paper import Paper

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

    nltk_tagger = NLTKTagger()

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

         # Remove dashes
         cleared_code_list[i] = cleared_code_list[i].replace("-", " ")


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
             word = item.text
             tag = item.tag_
             if tag != "VBG":
                 if tag == "NNS":
                     word = Word(word).singularize()
                 if "VB" in tag:
                     word = item.lemma_
                 tagged_words.append(word)

         for j in range(len(tagged_words)):
             #tagged_words[j] = remove_stopword(tagged_words[j])
             # Remove punctations
             tagged_words[j] = re.sub(r'([^a-zA-Z_]|_)+', '', tagged_words[j])

         #Remove redundant whitespaces
         cleared_code_list[i] = ' '.join(tagged_words)

         cleared_words = cleared_code_list[i].split()

         cleared_code_list[i] = ' '.join(cleared_words)

    final_codes = []
    final_codes.append(cleared_code_list)
    final_codes.append(original_codes)

    return final_codes

# Extract text from file and store its sentences in a list
def sent_tokenize_file(filename):

    list = convertEncoding(filename)
    word_list = brown.words()
    word_set = set(word_list)

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
    # Tokenize sentences
    sent_tokenize_list = sent_tokenize(fulltext)
    sent_tokenize_list_copy = sent_tokenize_list[:]

    cleared_list = []
    original_list = []

    for i in range(len(sent_tokenize_list)):

         # Expand contractions
         sent_tokenize_list[i] = expand_contractions(sent_tokenize_list[i])
         # Expand abbrevations
         sent_tokenize_list[i] = expand_abbrevations(sent_tokenize_list[i])
         # lowercasing
         sent_tokenize_list[i] = sent_tokenize_list[i].lower()

         words = sent_tokenize_list[i].split()

         for j in range(len(words)):

             # Join dash-seperated words if valid words are seperated
             if len(words[j]) > 0 and words[j][-1] == ("-") and j+1 < len(words):
                 if (re.sub(r'([^a-zA-Z_]|_)+', '', words[j]) + re.sub(r'([^a-zA-Z_]|_)+', '', words[j+1])) in word_set:
                     words[j] = words[j].replace("-", "") + words[j+1]
                     words[j+1] = ""
             if len(words[j]) > 0 and words[j][-1] == (u"\u2013") in words[j] and j+1 < len(words):
                 if (re.sub(r'([^a-zA-Z_]|_)+', '', words[j]) + re.sub(r'([^a-zA-Z_]|_)+', '', words[j + 1])) in word_set:
                    words[j] = words[j].replace(u"\u2013", "") + words[j+1]
                    words[j+1] = ""


         sent_tokenize_list[i] = ' '.join(words)

         # Replace dashes with whitespace
         sent_tokenize_list[i] = sent_tokenize_list[i].replace("-", " ")
         sent_tokenize_list[i] = sent_tokenize_list[i].replace(u"\u2013", " ")

         # Replace slashes with whitespace
         sent_tokenize_list[i] = sent_tokenize_list[i].replace("/", " ")

         sent_tokenize_list[i] = sent_tokenize_list[i].replace(u"\u2019", "")

         norm_word_list = []

         doc = nlp(sent_tokenize_list[i])

         for item in doc:
             word = item.text
             tag = item.tag_
             if tag == "NNS":
                word = Word(word).singularize()
             if "VB" in tag:
                word = item.lemma_
             norm_word_list.append(word)

         sent_tokenize_list[i] = ' '.join(norm_word_list)

         words = sent_tokenize_list[i].split()

         for j in range(len(words)):
             # Remove Stop Words
             #words[j] = remove_stopword(words[j])
             # Remove punctations and numbers
             words[j] = re.sub(r'([^a-zA-Z_]|_)+', '', words[j])
         sent_tokenize_list[i] = ' '.join(words)

         words = sent_tokenize_list[i].split()
         validWords = False
         letters = False
         for word in words:
             # Remove sententeces that contain no valid words
             if word in word_set:
                 validWords = True
             # Remove sententeces that contain letters only
             if len(word) > 1:
                 letters = True
         if validWords == True and letters == True and len(words)>1:
             cleared_list.append(' '.join(words))
             original_list.append(sent_tokenize_list_copy[i])

    final_list = []
    final_list.append(cleared_list)
    final_list.append(original_list)
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

def remove_stopword(s):

    if len(s) > 0 and (s[0].lower() + s[1:]) in stop_words:
        s = ""

    return s
start_time = time.time()
nlp = spacy.load('en', disable=['parser', 'ner', 'textcat'])
print(nlp.pipeline)
contractions_re = re.compile('(%s)' % '|'.join(CONTRACTIONS_DICT.keys()))
stop_words = set(stopwords.words('english'))

for line in read_codes(CODES_PATH):
    print(line)


def csvexport():

    #store paths of all main-texts in the given directory to a list
    filelist = iterate_folder(PAPER_DIR)


    # number of the paper
    i = 0
        #for line in sent_list[0]:
         #   print(line)
    with open('paper_export.csv','w') as f:
        # iterate through all main texts and print their sentences
        for file in filelist:
            i += 1
            print(i)
            sent_list = sent_tokenize_file(file)
            print(sent_list[0])
            print(sent_list[1])

            j = 0
            f.write("Paper_ID;PreProcessed;Original")
            f.write('\n')
            for item in sent_list[0]:
                f.write(sent_list[0][j].encode('utf-8') + ';'+ sent_list[1][j].encode('utf-8-sig'))
                f.write('\n')
                j = j + 1

def paper_to_list():

    filelist = iterate_folder(PAPER_DIR)

    # number of the paper
    i = 0

    preprocessed_paper_list = []

    # iterate through all main texts and print their sentences
    for file in filelist:
        i += 1
        print(i)
        sent_list = sent_tokenize_file(file)
        paper = Paper(i,sent_list[0],sent_list[1])
        preprocessed_paper_list.append(paper)

    return preprocessed_paper_list

print(paper_to_list())



# //store the sentences in a file
# writefile = io.open('S:\\VMs\\Shared\\Maindata.txt', 'w', encoding="utf-8-sig")
# for file in filelist:
#     i+=1
#     print(i)
#     for line in tokenize_file(file):
#         writefile.write(line + "\n")
#
# writefile.close()