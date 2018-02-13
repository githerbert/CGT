
# InferSent Model
echo 'Get InferSent Model'
(cd InferSent/encoder && curl -LO https://s3.amazonaws.com/senteval/infersent/infersent.allnli.pickle)

(cd InferSent/dataset && source get_data.bash)




