# -*- coding: utf-8 -*-
# import modules & set up logging
import os, io,re
from nltk.tokenize import sent_tokenize
from Definitions import ROOT_DIR, PAPER_DIR
from nltk.stem import WordNetLemmatizer
import fnmatch
from Contractions import CONTRACTIONS_DICT

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
    writefile = open(ROOT_DIR+'\\ConvertEncoding.txt', 'r+', encoding="utf-8-sig")
    writefile.truncate(0)
    readfile = open(filename, encoding="cp1252")
    for line in readfile:
        writefile.write(line)
        writefile.seek(0,0)
        list.append(writefile.read())
        writefile.truncate(0)
        writefile.seek(0, 0)

    readfile.close()
    writefile.close()

    return list

# Extract text from file and store its sentences in a list
def sent_tokenize_file(filename):

    list = convertEncoding(filename)
    wordnet_lemmatizer = WordNetLemmatizer()

    fulltext = ""

    for line in list:
        words = line.split()
    # Remove Headings
        if not all(word[0].isupper() for word in words) and words[-1] != ".":
            #Lemmatize words
            sentence = ' '.join([wordnet_lemmatizer.lemmatize(word) for word in words])
            #sentence = ' '.join([word for word in words])
            fulltext = ' '.join((fulltext, sentence))

    # Join dash-seperated words
    fulltext = fulltext.replace("- ", "")
    fulltext = fulltext.replace(u"\u2013 ", "")
    # Expand contractions
    fulltext = expand_contractions(fulltext)
    # Remove et al. from corpus
    fulltext = fulltext.replace(" et al.", "")
    sent_tokenize_list = sent_tokenize(fulltext)
    # Remove punctations
    # for i in range(len(sent_tokenize_list)):
    #     words = sent_tokenize_list[i].split()
    #     sent_tokenize_list[i] = ' '.join([re.sub(r'([^a-zA-Z0-9_]|_)+', '', word) for word in words])

    return sent_tokenize_list


def expand_contractions(s, contractions_dict=CONTRACTIONS_DICT):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

contractions_re = re.compile('(%s)' % '|'.join(CONTRACTIONS_DICT.keys()))

#store paths of all main-texts in the given directory to a list
filelist = iterate_folder(PAPER_DIR)


# number of the paper
i = 0

# iterate through all main texts and print their sentences
for file in filelist:
    i+=1
    print(i)
    for line in sent_tokenize_file(file):
        print(line)

# //store the sentences in a file
# writefile = io.open('S:\\VMs\\Shared\\Maindata.txt', 'w', encoding="utf-8-sig")
# for file in filelist:
#     i+=1
#     print(i)
#     for line in tokenize_file(file):
#         writefile.write(line + "\n")
#
# writefile.close()
