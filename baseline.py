import pyterrier as pt
import os
from helper import *

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
        return iter_indexer.index(load_collection(dataset_path, False, 2))
    else:
        print("Loading Index from disk...")
        return pt.IndexFactory.of(index_path)

def init_scorer(index):
    # Todo
    print("Initializing Scorer...")
    bm25 = pt.BatchRetrieve(index, wmodel="BM25")
    return bm25
    
if __name__ == "__main__":
    init_pyterrier()
    index_ref = init_indexer()
    bm_model = init_scorer(index_ref)
    
    ret = bm_model.search("post")
    print(ret)
