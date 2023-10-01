import pyterrier as pt
import os
from helper import *

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64/"


if __name__ == "__main__":
    ## <-- Initialize PyTerrier -->
    print("Running...")
    if not pt.started():
        pt.init()


    ## <-- Initalize The Indexer -->
    index_path = "./index"
    dataset_path = "datasets/collection.tsv"
    if not os.path.exists(index_path):
        print("Indexing documents...")
        iter_indexer = pt.IterDictIndexer(index_path)
        index_ref = iter_indexer.index(load_collection(dataset_path, False, 10000))
    else:
        print("Loading Index from disk...")
        index_ref = pt.IndexFactory.of(index_path)


    ## <-- Initalize The Scorer -->
    ## Todo
    print("Initializing Scorer...")
    ...
