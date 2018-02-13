infersent_model_path= 'https://s3.amazonaws.com/senteval/infersent/infersent.allnli.pickle'


# InferSent Model
echo $infersent_model_path
mkdir GloVe
curl -LO InferSent/encoder/infersent.allnli.pickle $infersent_model_path

source InferSent/dataset/get_data.bash




