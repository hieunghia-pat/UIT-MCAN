Deep Modular Co-Attention Network for ViVQA
----

This repository follows the paper [Deep Modular Co-Attention Networks for Visual Question Answering](https://arxiv.org/pdf/1906.10770.pdf) with modification to train on the [ViVQA dataset]() for VQA task in Vietnamese.

To reproduce the results on the ViVQA dataset, first you need to get the dataset as follow:
```
gdown --id 1TG2GQna8T7OPOuVw0RalO5WkafTgW6NT
unzip /content/ViVQA -d /content/ViVQA
```

After that, clone this repo locally then get into UIT-MCAN folder:
```
git clone https://github.com/hieunghia-pat/UIT-MCAN.git
cd UIT-MCAN
mkdir saved_models
```

Then download extracted image features:
```
gdown --id 1LIIcCsCjGUMyHlKi7lfojgah-kGr4V4a
```

Train the MCAN method with the following command:
```
python3 train.py
```

We especially design this method to train with Vietnamse pretrained word-embedding. To use pretrained word-embedding, open `config.py` then set word_embedding to the pretrained word-embedding you want:
```
"fasttext.vi.300d"
"phow2v.syllable.100d"
"phow2v.syllable.300d"
"phow2v.word.100d"
"phow2v.word.300d"
```