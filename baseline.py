import pyterrier as pt
import os
from typing import Dict, Generator

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64/"


def preprocess_text(text: str) -> str:
    # Todo
    ...


def load_collection(
    path: str, all_lines: bool = True, num_lines: int = 1
) -> Generator[Dict[str, str], None, None]:
    """
    Load documents from a TSV (Tab-Separated Values) file and yield them as dictionaries.

    Args:
        path (str): The file path to the TSV collection file.
        all_lines (bool, optional): If True, yield all lines in the file. If False,
            yield only every 'num_lines' lines (e.g., every 100 lines).
        num_lines (int, optional): The number of lines to skip when 'all_lines' is False.

    Yields:
        dict: A dictionary containing the loaded document's metadata.
            - 'docno' (str): The document identifier.
            - 'text' (str): The document's text content.

    Example:
        To load all documents from a TSV file:
        >>> for doc in load_collection("collection.tsv"):
        ...     print(doc)

        To load 100 documents from a TSV file:
        >>> for doc in load_collection("collection.tsv", all_lines=False, num_lines=100):
        ...     print(doc)
    """
    with open(path) as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            if index % num_lines == 0 and index != 0 and not all_lines:
                print(f"processing document {index}")
                break
            docno, text = line.rstrip().split("\t")
            yield {"docno": docno, "text": text}


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
