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
# Activate cuda
CUDA = True

# Show all paper details including single corresponding sentences in paper ranking
paper_details = True

infersent_threshold = 0.878
sent2vec_threshold = 0.847
