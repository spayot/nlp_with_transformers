# Chapter 9: Zero to Few Labels
Building a GitHub Issues Tagger

## How to Setup Environment
`make setup`  
`conda activate tfm9`
the above:  
* sets up a conda environment with necessary dependencies
* activates that environment
* pip installs the local package (`fewlabels`) in editable mode

## How to use
After setting up your environment, you can execute the notebook [here](notebooks/chp)

## Dataset
* [github issues tags for nlp with transformers]("https://git.io/nlp-with-transformers")
* 440 labeled issues + 9,303 unlabeled issues


## Main Topics Explored in that Chapter
* Multi-label classification
* Baseline: Naive Bayes cast as a one-vs-rest problem (using `scikit-multilearn`)
* zero-shot classfication: recasting problem as a text entailment problem. 
    * top k vs threshold approach
* Text data augmentation (`nlpaug` library) 
    * token perturbation / back-translation
* Using embeddings as a lookup table (FAISS)
    * how many neighbors? what threshold?
* Fine-Tuning a Vanilla Transformer
* Domain Adaptation: Fine-Tuning language model on unlabeled data before training model on labeled data.

![](images/)


## Refactored
* Training dataset slicing is refactored and extracted into `preproc.py`
* Evaluation Pipeline to get f1 micro/macro
* Cleaner functions to explore best label selection strategy based on zero shot classification scores (threshold vs top-k)
* Augmentation helper functions are refactored and moved to the `augment.py` module. defined several [callback protocols](https://mypy.readthedocs.io/en/stable/protocols.html) (interfaces) to decouple augmentation implementation from evaluation and batch level augmentation code.

## References:
* https://huggingface.co/docs/
