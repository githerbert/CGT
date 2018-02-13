import os,platform

# Papers path
PAPER_DIR = '/media/ubuntu/San480/VMs/Shared/Basket_Papers/Basket_Papers'
# Codes path
CODES_PATH = '/media/ubuntu/San480/VMs/Shared/Codes_Original.txt'

OS_NAME = platform.system()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Activate lemmatization
LEM = True
# Activate stopword removal
STOP = False
# Use Sent2Vec model (Needs 1,5 hours to process all papers)
SENT2VEC = False
# Use InferSent model (Needs 1 hour to process all papers with activated CUDA)
INFERSENT = True
# Activate cuda for inferSent (This is highly recommended because no CUDA use needs 10 hours)
CUDA = True

# Show all paper details including single corresponding sentences in paper ranking
paper_details = False

infersent_threshold = 0.878
sent2vec_threshold = 0.847
