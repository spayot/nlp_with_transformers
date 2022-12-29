# Chapter 7 - Question Answering
Setting up a Question Answering system using the haystack framework.

## Dataset: 
* in the book: subjqa - elecrtronics (customer reviews)
* here: subjqa - movies reviews
## Motivation
Allow to extract from previous reviews the most relevant answer to new free form text questions about a given item (here movie).

## Method and Results
Haystack uses a Retriever-Reader pipeline.
* the retriever allows to identify the most relevant reviews given a query
    * comparison between sparse (bm25) and dense (DRP) methods
* the reader extracts the passage in the document with the predicted answer to the question.
    * comparison between using a reader trained:
        * on SQUAD only  (100,000 question answer pairs)
        * on subjqa only (~1,300 question-answers pairs available for training)
        * on SQUAD & subjqa (domain adaptation)
    * as expected, best performance is achieved when using a model pre-trained on SQUAD and finetuned on subjqa

## How to Install
`setup.sh`

## How to use
after setting up your environment, you can execute the notebook [here](notebooks/chp7_question_answering.ipynb)

## Refactoring
* the code to convert training data into the SQUAD json format was refactored and extracted into 
## More Resources
* [subjqa dataset](https://huggingface.co/datasets/subjqa)
* [haystack documentation](https://docs.haystack.deepset.ai/docs/installation)
* [v2 of the books chp7 notebook](https://github.com/nlp-with-transformers/notebooks/blob/main/07_question_answering_v2.ipynb)

## About

