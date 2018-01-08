# -*- coding: utf-8 -*-
# import modules & set up logging
import os, io,re, nltk
from nltk.tokenize import sent_tokenize
from Definitions import ROOT_DIR, PAPER_DIR, CODES_PATH
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

# Extract codes from a file and store them in a list
def read_codes(filename):

    wordnet_lemmatizer = WordNetLemmatizer()
    codestring = ""

    readfile = open(filename, encoding='utf16')
    for line in readfile:

        words = line.split()

        if not all(word[0].isupper() for word in words) and words[-1] != ".":
            #Lemmatize words
            sentence = ' '.join([wordnet_lemmatizer.lemmatize(word) for word in words])
            codestring = ' <delimeter> '.join((codestring, sentence))

    # Remove dashes
    codestring = codestring.replace("-", " ")

    # Remove text between brackets
    codestring = re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", codestring)

    # Expand contractions
    codestring = expand_contractions(codestring)
    # Remove et al. from corpus
    codestring = codestring.replace(" et al.", "")
    # To lower case
    codestring = codestring.lower()
    # Tokenize codes
    tokenized_codes = codestring.split(' <delimeter> ')
    # Remove punctations
    for i in range(len(tokenized_codes)):
         words = tokenized_codes[i].split()

         #Remove verbs at the beginning of the codes
         tagged_words = []

         for item in words:
             tokenized = nltk.word_tokenize(item)
             tagged = nltk.pos_tag(tokenized)
             if tagged[0][1] != "VBG":
                 tagged_words.append(tagged[0][0])

         tokenized_codes[i] = ' '.join([re.sub(r'([^a-zA-Z_]|_)+', '', word) for word in tagged_words])

    return tokenized_codes

# Extract text from file and store its sentences in a list
def sent_tokenize_file(filename):

    list = convertEncoding(filename)
    wordnet_lemmatizer = WordNetLemmatizer()

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
    # To lower case
    fulltext = fulltext.lower()
    # Tokenize sentences
    sent_tokenize_list = sent_tokenize(fulltext)
    # Remove punctations
    for i in range(len(sent_tokenize_list)):
         words = sent_tokenize_list[i].split()
         sent_tokenize_list[i] = ' '.join([re.sub(r'([^a-zA-Z_]|_)+', '', word) for word in words])

    return sent_tokenize_list


def expand_contractions(s, contractions_dict=CONTRACTIONS_DICT):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

contractions_re = re.compile('(%s)' % '|'.join(CONTRACTIONS_DICT.keys()))

for line in read_codes(CODES_PATH):
    print(line)

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
