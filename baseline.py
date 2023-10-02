import pyterrier as pt
import os
from helper import *
import pandas as pd
from nltk.stem import WordNetLemmatizer
import logging
logging.basicConfig(filename='baseline.log', format='%(asctime)s %(message)s', level=logging.INFO)
## os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64/"


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

def init_scorer_bm25(index):
    # Todo
    print("Initializing Scorer...")
    bm25 = pt.BatchRetrieve(index, wmodel="BM25")
    return bm25

def init_scorer_tf_idf(index):
    # Todo
    tf_idf = pt.BatchRetrieve(index, wmodel="TF_IDF")
    return tf_idf


def run_queries(bm_model, index):
    queries_path = "datasets/queries_train.csv"
    qrels_path = "datasets/qrels_train.txt"
    topics = pd.read_csv(queries_path)
    topics = topics[["qid", "query"]]
    topics["query"] = topics["query"].apply(lambda x: preprocess_text(x))
    topics["query"] = topics["query"].apply(lambda x: " ".join([stemmer.stem(word) for word in x]))

    qrels = pt.io.read_qrels(qrels_path)
    res = bm_model.transform(topics)
    eval = pt.Utils.evaluate(res, qrels, metrics = ['map'])
    print(eval)


def eval_queries():
    pass


def query_experiment(bm25, tf_idf):
    lemmatizer = WordNetLemmatizer()
    queries_path = "datasets/queries_train.csv"
    qrels_path = "datasets/qrels_train.txt"
    topics = pd.read_csv(queries_path)
    topics = topics[["qid", "query"]]
    topics["query"] = topics["query"].apply(lambda x: preprocess_text(x, lemmatizer))
    qrels = pt.io.read_qrels(qrels_path)

    tf_idf = tf_idf.transform(topics)
    bm25 = bm25.transform(topics)
    res = pt.Experiment(
        [tf_idf, bm25],
        topics,
        qrels,
        eval_metrics=["map", "recip_rank"],
        names=["TF_IDF", "BM25"]
    )
    print(res)

if __name__ == "__main__":
    init_pyterrier()
    index = init_indexer()
    #logging.info(index.getCollectionStatistics())
    #for kv in index.getLexicon():
    #    print((kv.getKey()))

    bm_model = init_scorer_bm25(index)
    tf_idf = init_scorer_tf_idf(index)
    query_experiment(bm_model, tf_idf)
