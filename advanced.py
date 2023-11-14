import pyterrier as pt
import os
from helper import *
import pandas as pd
from nltk.stem import WordNetLemmatizer
from pyterrier.measures import Recall, AP, RR, nDCG
import logging
from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker

logging.basicConfig(filename='advanced.log', format='%(asctime)s %(message)s', level=logging.INFO)

def init_pyterrier():
    print("Running...")
    if not pt.started():
        pt.init()

def init_indexer():
    index_path = "./index_advanced"
    dataset_path = "datasets/collection.tsv"
    if not os.path.exists(index_path):
        print("Indexing documents...")
        iter_indexer = pt.IterDictIndexer(index_path)
        iter_indexer.index(load_collection(dataset_path), meta={'docno' : 20, 'text': 4096})
    else:
        print("Loading Index from disk...")
    return pt.IndexFactory.of(index_path)

def init_scorer(index, model):
    print(f"Initializing Scorer: {model}...")
    pass

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
    index = init_indexer()
    logging.info(index.getCollectionStatistics())

    lemmatizer = WordNetLemmatizer()
    queries_path = "datasets/queries_train.csv"
    topics = pd.read_csv(queries_path)
    topics = topics[["qid", "query"]]
    topics["query"] = topics["query"].apply(lambda x: preprocess_text(x, lemmatizer))
    qrels_path = "datasets/qrels_train.txt"
    qrels = pt.io.read_qrels(qrels_path)

    monoT5 = MonoT5ReRanker()
    duoT5 = DuoT5ReRanker()
    bm25 = pt.BatchRetrieve(index, wmodel="BM25") % 100
    mono_pipeline = bm25 >> pt.text.get_text(index, "text") >> monoT5
    duo_pipeline = mono_pipeline % 10 >> duoT5

    result = result = duo_pipeline.transform(topics)

    chunk_size = 10
    num_chunks = (len(result) // chunk_size) + 1
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk = result.iloc[start_idx:end_idx]
        sorted_chunk = chunk.sort_values(by='rank')
        result.iloc[start_idx:end_idx] = sorted_chunk