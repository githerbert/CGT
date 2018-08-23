## Computational Grounded Theory

This application basically tries to solve the ranking problem of information retrieval. For given query it searches through given documents for semantic matches and creates a ranking of the relevance of documents concerning the query. Originally the main purpose of this application is to support researches when they are conducting research with Grounded Theory methodology. The manual expensive literature search should be computerised that the methodology become Computational Grounded Theory. However, the application can also be used for any Text search where you want so to find semantic matches of query and document. The requirement is that the query and document must be in .txt-format. Furthermore, the sentences in a document must be separated by points (.).
In Grounded Theory terminology the query is called “Code” and the documents are called scientific papers. Therefore you will only find the denotation “Code” and “papers” in the source code.
A semantic search procedure includes four steps: Pre-Processing, Sentence Encoding, Sentence comparison and paper ranking.
The query and the documents are pre-processed which includes sentence tokenizing, lemmatisation and stop word removal (Last two are optional). The document pre-processing is fitted to papers that belong to the “Senior Scholars' Basket of Journals” (https://aisnet.org/page/SeniorScholarBasket), the top 8 journals of the Information System domain.
During Sentence Encoding the sentences of query and documents is converted to vectors in a vector space model. The user has the choice to switch between sent2vec and InferSent sentence encoders. The first uses Bag of Words approach where the word order isn’t considered while the latter uses Recurrent Neural Network (Bidirectional LSTM) and treats sentences like a sequence of words where word order does play a role.
The sentence comparison is realised through cosine-distance. A document sentence matches with a query sentence when a certain threshold is reached. The predefined thresholds in definition.py stem from human recognition regarding relevance in the Basket of Eight.
To create the Paper ranking the matches for every document are summed up. The final output is a list of the 10 most relevant documents for a query in descending order. 
To use this application Linux or macOS is required. In addition, you need Python 2.7 installed.

## Prerequisites

 - Linux or MacOS
 - Python 2.7
 
## Installation

 Get the repository 


    git clone https://github.com/simonhess/CGT.git
	
 Install the following Python libraries:

	1. Spacy
		conda install -c conda-forge spacy
	2. Spacy english
		python -m spacy download en
	3. nltk
		conda install -c anaconda nltk
	4. nltk punkt
		python import nltk
		python nltk.download('punkt')
	5. prettytable
		conda install -c synthicity prettytable 
	6. numpy
		conda install -c anaconda numpy
	7. SciPy
		conda install -c anaconda scipy
	8. unicodecsv
		conda install -c anaconda unicodecsv
	9. regex
		conda install -c conda-forge regex

### Depending on the sentence enconder you want to use further installation are needed
		
### Sent2Vec Install Instruction

1. change directory to CGT/Sent2Vec
2. use "make" command so C++ files are compiled (Requires C++ compiler like g++)
3. download the pre-trained model (2GB):
https://drive.google.com/open?id=0B6VhzidiLvjSOWdGM0tOX1lUNEk
4. move the downloaded torontobooks_unigrams.bin file to path CGT/Sent2Vec

### InferSent Install Instruction

1. Install Nvidia CUDA (Optional but highly recommended)

2. Install pytorch by:

	conda install -c soumith pytorch

3. Run script in CGT folder to download pre-trained model and GloVe vectors (2,5GB):

	source get_data.bash
	
### Set parameters and start the programm
		
 Set parameters in CGT/Definitions.py including the variables:
	- PAPER_DIR: Path of the papers folder
	- CODES_PATH: Path to the code .txt- file (There must be one code per line)
	- LEM: Activation of Lemmatization
	- STOP: Activation of StopWord Removal
	- SENT2VEC/INFERSENT: Selection of Model (Sent2Vec or InferSent)
	- CUDA: Activation of CUDA (If InferSent is used)
	- paper_details: Activate to show all corresponding sentences to a paper in paper ranking

 Optionally domain specific abbreviations can be added to the ABBREVATIONS_DICT dictionary in the preprocessing_lib.py file

 If all parameters are set correctly the program can be used by running the main file in CGT folder:
	python main.py
 (This needs up to 2,5 hours for all Basket of Eight papers)

	
