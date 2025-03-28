# Usage
Trigrams’n’Tags (TnT) is an efficient statistical
[part-of-speech tagger](https://universaldependencies.org/u/pos/). This project implements the Hidden Markoc Model with a beam search to reduce the processing time from the paper TnT - A Statistical Part-of-Speech Taggerby Thorsten Brants [1].

### Run from CLI
To use it, simply run or pass your own sentence to tag.
```
python tnttagger --sentence "this is a test sentence"
```

Output:
```
Sentence: ['this', 'is', 'a', 'test', 'sentence']
POS Tags: [('this', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('test', 'NN'), ('sentence', 'NN')]
```

Works pretty well for a model purely based on statistical information, you can also pass the dataset you want to train/populate, default is `treebanck`, replace with other nltk dataset using `--traindata 'brown'`. 

Few changes from standard implementation is that it doesn't deal with unkown words and tags them as unk, and uses a simpler conditional frequency recorder class SimpleCDF for populating the tags.

## Reference
[1] Brants, T. (2000). TnT – A Statistical Part-of-Speech Tagger. [arXiv:cs/0003055.](https://arxiv.org/abs/cs/0003055)
