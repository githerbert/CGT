# -*- coding: utf-8 -*-
# import modules & set up logging
import os, io,re, nltk
from nltk.tokenize import sent_tokenize
from Definitions import ROOT_DIR, PAPER_DIR, CODES_PATH, OS_NAME
from nltk.corpus import brown, stopwords
import fnmatch
from Contractions import CONTRACTIONS_DICT
from Abbrevations import ABBREVATIONS_DICT

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

         # Remove dashes
         cleared_code_list[i] = cleared_code_list[i].replace("-", " ")

         # Remove text between brackets
         cleared_code_list[i] = re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", cleared_code_list[i])

         # Expand contractions
         cleared_code_list[i] = expand_contractions(cleared_code_list[i])

         words = cleared_code_list[i].split()

         #Remove verbs at the beginning of the codes
         tagged_words = []

         for item in words:
             tokenized = nltk.word_tokenize(item)
             tagged = nltk.pos_tag(tokenized)
             if tagged[0][1] != "VBG":
                 tagged_words.append(tagged[0][0])

         cleared_words=[]

         for j in range(len(tagged_words)):
             tagged_words[j] = remove_stopword(tagged_words[j])
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
         words = sent_tokenize_list[i].split()
         for j in range(len(words)):
             # Remove Stop Words
             words[j] = remove_stopword(words[j])
             # Join dash-seperated words
             words[j] = words[j].replace("- ", "")
             words[j] = words[j].replace(u"\u2013 ", "")
             # Expand contractions
             words[j] = expand_contractions(words[j])
             # Expand abbrevations
             words[j] = expand_abbrevations(words[j])
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

    if (s[0].lower() + s[1:]) in stop_words:
        s = ""

    return s

contractions_re = re.compile('(%s)' % '|'.join(CONTRACTIONS_DICT.keys()))
stop_words = set(stopwords.words('english'))

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
    sent_list = sent_tokenize_file(file)
    print(sent_list[0])
    print(sent_list[1])
    #for line in sent_list[0]:
     #   print(line)

# //store the sentences in a file
# writefile = io.open('S:\\VMs\\Shared\\Maindata.txt', 'w', encoding="utf-8-sig")
# for file in filelist:
#     i+=1
#     print(i)
#     for line in tokenize_file(file):
#         writefile.write(line + "\n")
#
# writefile.close()
