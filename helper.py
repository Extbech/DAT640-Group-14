from typing import Dict, Generator

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
