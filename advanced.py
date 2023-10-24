import pyterrier as pt
import os
from helper import *
import pandas as pd
from nltk.stem import WordNetLemmatizer
from pyterrier.measures import Recall, AP, RR, nDCG
import logging

logging.basicConfig(filename='advanced.log', format='%(asctime)s %(message)s', level=logging.INFO)

from sklearn.ensemble import RandomForestRegressor

def init_pyterrier():
    print("Running...")
    if not pt.started():
        pt.init()


def init_scorer(index):
    lemmatizer = WordNetLemmatizer()
    ## Load Train Topics
    queries_path = "datasets/queries_train.csv"
    topics = pd.read_csv(queries_path)
    topics = topics[["qid", "query"]]
    topics["query"] = topics["query"].apply(lambda x: preprocess_text(x, lemmatizer))

    ## Load Qrels
    qrels_path = "datasets/qrels_train.txt"
    qrels = pt.io.read_qrels(qrels_path)

    rf = RandomForestRegressor(n_estimators=400)
    bm25 = pt.BatchRetrieve(index, wmodel="BM25", controls={"bm25.k1": "1", "bm25.b": "0.5"})
    tf = pt.BatchRetrieve(index, wmodel="Tf")
    pl2 = pt.BatchRetrieve(index, wmodel="PL2")
    pipeline = bm25 >> (tf ** pl2)
    rf_pipe = pipeline >> pt.ltr.apply_learned_model(rf)
    rf_pipe.fit(topics, qrels)
    return pt.Experiment([bm25, rf_pipe], topics, qrels, eval_metrics = ['map', "recip_rank", Recall@1000, AP(rel=2), RR(rel=2), nDCG@3], names=["BM25 Baseline", "LTR"])


def index_colbert(dataset_path, index_path):
    indexer = ColBERTIndexer("http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip", index_path, "index_colbert", chunksize=3)
    indexer.index(load_collection(dataset_path))

def score_queries(model):
    lemmatizer = WordNetLemmatizer()
    queries_path = "datasets/queries_train.csv"
    topics = pd.read_csv(queries_path)
    topics = topics[["qid", "query"]]
    topics["query"] = topics["query"].apply(lambda x: preprocess_text(x, lemmatizer))

    res = model.transform(topics)
    create_dir_if_not_exists("results")
    print(f"Writing Query Results to results/trec_result_{model}.txt...")
    pt.io.write_results(res, f"results/trec_result_{model}.txt", format="trec", run_name="BM25")
    return res


def evaluate_result(result):
    qrels_path = "datasets/qrels_train.txt"
    qrels = pt.io.read_qrels(qrels_path)
    eval = pt.Utils.evaluate(result, qrels, metrics = ['map', "recip_rank", Recall@1000, AP(rel=2), RR(rel=2), nDCG@3])
    return eval



if __name__ == "__main__":
    init_pyterrier()
    from pyterrier_colbert.indexing import ColBERTIndexer
    import faiss
    index_colbert("datasets/collection.tsv", "./colbert_index")
