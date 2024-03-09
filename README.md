# NLPproject-Fact_checked-claims-detect
Delivered as a project for the course "Natural Language Processing"


-------
NLP PROJECT done by Michele Luca Contalbo, Yiran Zeng.

The ipynb file was run on colab, so it may have instructions specific to that platform.
The directory is composed in the following way:
- train,dev and vclaim datasets
- analysis, which contains some saved data (mainly the one used for plots) and co2 emissions data.
  The emissions data file has 4 rows, corresponding to bm25, tf-idf, sbert and sgpt
- scorer, which has the necessary functions to import to evaluate the models

In the code, we willingly left some code cells used to store/load variables or other objects with pickle,
so that data can easily be recovered or dumped.

To run the training procedure on sentence_transformers model, the SentenceTransformer.py file in this directory
must be loaded in the colab runtime (replacing the standard SentenceTransformer.py).
This new version takes validation data and prints train and val error for each epoch.
