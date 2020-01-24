# Named Entity Relation
In this notebook I train/implement a NER model using Conditional Random fields in pytorch. The CRF layer is compeletely self implemented using forward algorithm and to decode the viterbi decoding is applied using back pointers, also self implemented. The model is as given in this [paper](https://arxiv.org/pdf/1603.01360.pdf)

# Dataset
 I use the NER data set available on [Kaggle](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus). This is basically just a map between each word in a sentence to a tag.<br />
You can find out about the tagging scheme [here](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging))


# Pre-processing
 Firstly I extract all sentences in the format
 ```
 word-->tag
 ```
 Now extract all the unique words and tags and give them a unique index. Then store maps word2index , tag2index and vice-versa. Three new tags are introduced...
 ```
 <PAD> , <BOS> and <EOT>
 ```
 
 # 
 
 
