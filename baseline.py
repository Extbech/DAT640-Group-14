import pyterrier as pt
import os
from helper import *
import pandas as pd
from nltk.stem import WordNetLemmatizer
from pyterrier.measures import Recall, AP, RR, nDCG
import logging
logging.basicConfig(filename='baseline.log', format='%(asctime)s %(message)s', level=logging.INFO)


def init_pyterrier():
    print("Running...")
    if not pt.started():
        pt.init()

def init_indexer():
    index_path = "./index"
    dataset_path = "datasets/collection.tsv"
    if not os.path.exists(index_path):
        print("Indexing documents...")
        iter_indexer = pt.IterDictIndexer(index_path)
        iter_indexer.index(load_collection(dataset_path))
    else:
        print("Loading Index from disk...")
    return pt.IndexFactory.of(index_path)

def init_scorer(index, model):
    print(f"Initializing Scorer: {model}...")
    bm25 = pt.BatchRetrieve(index, wmodel=model)
    return bm25

def score_queries(model):
    lemmatizer = WordNetLemmatizer()
    queries_path = "datasets/queries_train.csv"
    topics = pd.read_csv(queries_path)
    topics = topics[["qid", "query"]]
    topics["query"] = topics["query"].apply(lambda x: preprocess_text(x, lemmatizer))

    res = model.transform(topics)
    pt.io.write_results(res, f"results/trec_result_{model}.txt", format="trec")
    return res

def evaluate_result(result):
    qrels_path = "datasets/qrels_train.txt"
    qrels = pt.io.read_qrels(qrels_path)
    eval = pt.Utils.evaluate(result, qrels, metrics = ['map', Recall@1000, AP(rel=2), RR(rel=2), nDCG@3])
    return eval

if __name__ == "__main__":
    init_pyterrier()
    index = init_indexer()
    logging.info(index.getCollectionStatistics())

    bm_model = init_scorer(index, "BM25")
    tf_idf = init_scorer(index, "TF_IDF")

    print("BM25 results...")
    result_bm = score_queries(bm_model)
    eval_bm = evaluate_result(result_bm)
    print(eval_bm)

    print("TF_IDF results...")
    result_tf_idf = score_queries(tf_idf)
    eval_tf_idf = evaluate_result(result_tf_idf)
    print(eval_tf_idf)
