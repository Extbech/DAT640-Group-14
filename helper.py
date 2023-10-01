from typing import Dict, Generator
import re
import nltk
nltk.download("stopwords")
STOPWORDS = set(nltk.corpus.stopwords.words("english"))


def preprocess_text(text: str) -> str:
    """Preprocesses a string of text.

    Args:
        text: A string of text.

    Returns:
        preprocessed string.
    """
    return " ".join([
        term
        for term in re.sub(r"[^\w]|_", " ", text).lower().split()
        if term not in STOPWORDS
    ])

def load_collection(
    path: str, all_lines: bool = True, num_lines: int = 1
) -> Generator[Dict[str, str], None, None]:
    """
    Load documents from a TSV (Tab-Separated Values) file and yield them as dictionaries.

    Args:
        path (str): The file path to the TSV collection file.
        all_lines (bool, optional): If True, yield all lines in the file. If False,
            yield 'num_lines' lines (e.g., 100 lines).
        num_lines (int, optional): The number of lines to skip when 'all_lines' is False.
    
    If no optional arguments are supplied, default behaviour is to load the entire file.

    Yields:
        dict: A dictionary containing the loaded document's metadata.
            - 'docno' (str): The document identifier.
            - 'text' (str): The document's text content.
    """
    with open(path) as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            if index % num_lines == 0 and index != 0 and not all_lines:
                print(f"processing document {index}")
                break
            docno, text = line.rstrip().split("\t")
            yield {"docno": docno, "text": preprocess_text(text)}
