import pyterrier as pt
import os
from helper import *
from pyterrier.measures import Recall, AP, RR, nDCG, MRR
import logging
from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(filename='advanced.log', format='%(asctime)s %(message)s', level=logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def init_pyterrier():
    print("Running...")
    if not pt.started():
        # pt.init()
        pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

def init_index_baseline():
    index_path = "./index_baseline"
    if not os.path.exists(index_path):
        dataset_path = "datasets/collection.tsv"
        iter_indexer = pt.IterDictIndexer(index_path, blocks=True, meta={'docno' : 20, 'text': 4096})
        iter_indexer.index(load_collection(dataset_path))
    return pt.IndexFactory.of(index_path)

def init_index_expanded():
    index_path = "./index_expanded"
    if not os.path.exists(index_path):
        dataset_path = "datasets/collection_expanded.tsv"
        iter_indexer = pt.IterDictIndexer(index_path, blocks=True)
        iter_indexer.index(load_collection(dataset_path))
    return pt.IndexFactory.of(index_path)

def context_rewrite(topics, tok, model_rewrite):
    return_queries = []
    for _, entry in topics.iterrows():
        conversation_history = topics[(topics['topic_number'] == entry['topic_number']) & (topics['turn_number'] < entry['turn_number'])]
        context_input = "|||".join(conversation_history['query'].tolist() + [entry['query']])
        input_ids = tok.encode(context_input, return_tensors="pt")
        outputs = model_rewrite.generate(input_ids, max_length=512)
        query_rewrite = tok.decode(outputs[0], skip_special_tokens=True)
        return_queries.append(preprocess_text(query_rewrite))
    return return_queries

def run_mono_duo(topics, mono_reranking=1000, duo_reranking=50):
    index_baseline = init_index_baseline()
    index_expanded = init_index_expanded()
    monoT5 = MonoT5ReRanker(batch_size=4)
    duoT5 = DuoT5ReRanker(batch_size=4)
    bm25_1 = pt.BatchRetrieve(index_expanded, wmodel="BM25", verbose=True, num_results=1000, controls={"bm25.k1": "0.82", "bm25.b": "0.68"})
    tok = AutoTokenizer.from_pretrained("castorini/t5-base-canard")
    model_rewrite = AutoModelForSeq2SeqLM.from_pretrained("castorini/t5-base-canard")
    topics["query"] = context_rewrite(topics, tok, model_rewrite)
    pipeline = bm25_1 % mono_reranking >> pt.text.get_text(index_baseline, "text") >> monoT5 % duo_reranking >> duoT5
    return pipeline.transform(topics[["qid", "query"]])

def evaluate_result(result):
    qrels_path = "datasets/qrels_train.txt"
    qrels = pt.io.read_qrels(qrels_path)
    eval = pt.Utils.evaluate(result, qrels, metrics = ['map', "recip_rank", Recall@1000, AP(rel=2), RR(rel=2), nDCG@3])
    return eval

if __name__ == "__main__":
    init_pyterrier()
    
    set = "test"
    topics = load_queries(set)
    mono_reranking = 50
    duo_reranking = 10
    result = run_mono_duo(topics, mono_reranking, duo_reranking)

    sort_result(result, duo_reranking)
    
    if set == "train":
        print(evaluate_result(result))

    submission_name = "expandoMonoDuo"
    pt.io.write_results(result, f"results/trec_result_{set}_{submission_name}.txt", format="trec", run_name=submission_name)
    if set == "test":
        create_submission(submission_name, duo_reranking)