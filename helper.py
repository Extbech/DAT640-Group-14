import os
from typing import Dict, Generator
import re
import nltk
import pandas as pd
import pyterrier as pt
nltk.download("stopwords")
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
STOPWORDS = set(nltk.corpus.stopwords.words("english"))


def preprocess_text(text: str) -> str:
    """Preprocesses a string of text.

    Args:
        text: A string of text.

    Returns:
        preprocessed string.
    """
    return " ".join([term for term in re.sub(r"[^\w]|_", " ", text).lower().split()])

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
        for index, line in enumerate(f):
            if index % num_lines == 0 and index != 0 and not all_lines:
                print(f"processing document {index}")
                break
            docno, text = line.rstrip().split("\t")
            yield {"docno": docno, "text": text}

def load_queries(
        set: str = "train"
) -> pd.DataFrame:
    """
    Loads queries to be used for scoring.

    Args:
        set (str): The set of queries to be used. Either "train" or "test".

    Returns:
        pd.DataFrame: DataFrame with "qid" and "query" columns.
    """
    queries_path = f"datasets/queries_{set}.csv"
    topics = pd.read_csv(queries_path)
    topics = topics[["qid", "query"]]
    topics["query"] = topics["query"].apply(lambda x: preprocess_text(x))
    return topics

def sort_result(
        result: pd.DataFrame,
        number_of_rankings: int
) -> None:
    """
    For some reason, the monoDuo does not return the results dataframe sorted by rank.
    This code does an inplace sorting on the results dataframe.

    Args:
        result (pd.DataFrame): The results dataframe to sort.
        number_of_rankings (int): The number of rankings per query
    """
    num_chunks = (len(result) // number_of_rankings) + 1
    for i in range(num_chunks):
        start_idx = i * number_of_rankings
        end_idx = (i + 1) * number_of_rankings
        chunk = result.iloc[start_idx:end_idx]
        sorted_chunk = chunk.sort_values(by='rank')
        result.iloc[start_idx:end_idx] = sorted_chunk

def create_submission(
    submission_name: str,
    number_of_rankings: int
) -> None:
    """
    Creates a .csv file in the format needed to submit the solution to Kaggle.

    Args:
        submission_name (str): Name of the submission.
        number_of_rankings (int): Number of rankings per query in the result.
    """
    results = pt.io.read_results(f"./results/trec_result_test_{submission_name}.txt", format="trec")
    results['group'] = [i // number_of_rankings for i in range(len(results))]
    results = results.groupby('group').head(3).reset_index(drop=True)
    results = results.drop('group', axis=1)
    results.rename(columns={"docno": "docid"}, inplace=True)
    results = results[["qid", "docid"]]
    
    results.to_csv(f"./results/trec_result_test_{submission_name}.csv", index=None)

def create_dir_if_not_exists(dir_name: str) -> None:
    """
    Create a directory if it does not already exist.

    Args:
        dir_name (str): The name of the directory to be created.

    Returns:
        None: No return value.
    """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        
