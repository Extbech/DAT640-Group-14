import pyterrier as pt
import os
from helper import *
from pyterrier.measures import Recall, AP, RR, nDCG
import logging
from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker
from transformers import AutoTokenizer

logging.basicConfig(filename='advanced.log', format='%(asctime)s %(message)s', level=logging.INFO)

def init_pyterrier():
    print("Running...")
    if not pt.started():
        # pt.init()
        pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

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

def run_mono_duo(mono_reranking=1000, duo_reranking=50):
    monoT5 = MonoT5ReRanker()
    duoT5 = DuoT5ReRanker()
    bm25 = pt.BatchRetrieve(index, wmodel="BM25", controls={"bm25.k1": "0.82", "bm25.b": "0.68", "qe": "on", "qemodel": "Bo1"})
    mono_pipeline = bm25 % mono_reranking >> pt.text.get_text(index, "text") >> monoT5
    duo_pipeline = mono_pipeline % duo_reranking >> duoT5
    return duo_pipeline.transform(topics)

def evaluate_result(result):
    qrels_path = "datasets/qrels_train.txt"
    qrels = pt.io.read_qrels(qrels_path)
    eval = pt.Utils.evaluate(result, qrels, metrics = ['map', "recip_rank", Recall@1000, AP(rel=2), RR(rel=2), nDCG@3])
    return eval

if __name__ == "__main__":
    init_pyterrier()
    index = init_indexer()
    logging.info(index.getCollectionStatistics())

    set = "train"
    topics = load_queries(set)
    mono_reranking = 100
    duo_reranking = 10
    result = run_mono_duo(mono_reranking, duo_reranking)
    sort_result(result, duo_reranking)
    
    if set == "train":
        print(evaluate_result(result))

    submission_name = "monoDuo"
    pt.io.write_results(result, f"results/trec_result_{set}_{submission_name}.txt", format="trec", run_name=submission_name)
    if set == "test":
        create_submission(submission_name, duo_reranking)