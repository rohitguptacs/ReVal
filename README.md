#######################################################
ReVal: A Simple and Effective Machine Translation Evaluation Metric Based on Recurrent Neural Networks
#######################################################

Please refer to the following [paper](http://pers-www.wlv.ac.uk/~in4089/publications/2015/guptaemnlp2015.pdf) for details about this metric and the generation of training data:
Rohit Gupta, Constantin Orasan, and Josef van Genabith. 2015. [ReVal: A Simple and Effective Machine Translation Evaluation Metric Based on Recurrent Neural Networks](http://pers-www.wlv.ac.uk/~in4089/publications/2015/guptaemnlp2015.pdf). In Proceedings of the Conference on Empirical Methods in Natural Language Processing, EMNLP ’15, Lisbon, Portugal.

This code is available at [GitHub](https://github.com/rohitguptacs/ReVal).

The TreeStructured-LSTM code used in this metric implementation is obtained from https://github.com/stanfordnlp/treelstm.[Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](http://arxiv.org/abs/1503.00075) by Kai Sheng Tai, Richard Socher, and Christopher Manning.


#####################Installation and Running###################################
## Software Requirements:
- Java >= 8 (for Stanford CoreNLP utilities)
- Python >= 2.7
If you do not have lua you need to install [lua](http://www.lua.org/download.html) and [luarocks](https://luarocks.org/#quick-start) first and install the following: 
- [Torch7](https://github.com/torch/torch7)
- [penlight](https://github.com/stevedonovan/Penlight)
- [nn](https://github.com/torch/nn)
- [nngraph](https://github.com/torch/nngraph)
- [optim](https://github.com/torch/optim)
by using [luarocks](https://luarocks.org/#quick-start)

```
For example:
luarocks install nngraph
```

## Running the meric

First download the required data and libraries by running the following script:

```
./download_and_preprocess.sh
```

This will download the Glove vectors, Stanford Parser and POS tagger:
  - [Glove word vectors](http://nlp.stanford.edu/projects/glove/) (Common Crawl 840B) -- **Warning:** this is a 2GB download!
  - [Stanford Parser](http://nlp.stanford.edu/software/lex-parser.shtml)
  - [Stanford POS Tagger](http://nlp.stanford.edu/software/tagger.shtml)


To run the metric (Currently this metric only evaluates translations into English)
run:

```
python ReVal.py -r sample_reference.txt -t sample_translation.txt

```
replace sample_reference.txt and sample_translation.txt with your reference and translation files

## Training the metric

For training the metric you need to [download the training data](https://www.dropbox.com/s/wk10mhajytf1uj1/WMT13SetL.zip?dl=0). If you plan to replicate all the resluts given in [paper](http://pers-www.wlv.ac.uk/~in4089/publications/2015/guptaemnlp2015.pdf) you will also need [SICK](http://alt.qcri.org/semeval2014/task1/data/uploads/sick_train.zip) data.

Preprocess (use -h for help) and train using:
``` 
python scripts/preprocess-training-data.py -t training/qsetl_train.txt -d training/qsetl_dev.txt
th relatedness/trainingscript.lua  --dim <LSTM_memory_dimension(default:150)> --epochs <number_of_training_epochs(default:10)>

```


