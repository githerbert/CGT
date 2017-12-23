# -*- coding: utf-8 -*-
# import modules & set up logging
import os, io
from nltk.tokenize import sent_tokenize
from Definitions import ROOT_DIR
import fnmatch

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()



def iterate_folder(root):

    rootdir = root
    i = 0

    mainfiles = []

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if fnmatch.fnmatch(file, 'Main.txt'):
                mainfiles.append(os.path.join(subdir, file))
    return mainfiles

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

# Extract text from file and store it in a list
def tokenize_file(filename):

    list = convertEncoding(filename)

    fulltext = ""

    for line in list:
        words = line.split()
        sentence = ' '.join([word for word in words])
        fulltext = ' '.join((fulltext, sentence))

    # Join dash-seperated words
    fulltext = fulltext.replace("- ", "")
    fulltext = fulltext.replace(u"\u2013 ", "")
    # Remove et al. from corpus
    fulltext = fulltext.replace(" et al.", "")
    sent_tokenize_list = sent_tokenize(fulltext)
    print(sent_tokenize_list[1])

    return sent_tokenize_list


filelist = iterate_folder('S:\\VMs\\Shared\\Basket_Papers\\Basket_Papers\\misq\\2017\\3\\1')



i = 0

writefile = io.open('S:\\VMs\\Shared\\Maindata.txt', 'w', encoding="utf-8-sig")
for file in filelist:
    i+=1
    print(i)
    for line in tokenize_file(file):
        writefile.write(line + "\n")

writefile.close()

