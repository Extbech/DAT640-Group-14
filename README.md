# DAT640 Project Work
## Introduction
This repository contains the code for a project focused on indexing a large collection of documents and creating a scorer that can receive a query and return the top 1000 documents that best match the given query. The project primarily utilizes Python and the PyTerrier library for indexing and information retrieval tasks.

## Getting Started
### Prerequisites
Before you begin, make sure you have the following installed:

- Python
- PyTerrier
- Java (Java 11 or higher)
You can set the JAVA_HOME environment variable to point to your Java installation:

```bash
export JAVA_HOME="<path>"
```
To install the required dependancies and libraries you can run the following command:
```bash
pip install -r requirements.txt
```

### Dataset
Due to the large size of the dataset, it is not hosted on GitHub. To use the project, you will need to provide your dataset and place it in a folder called `/datasets`. Files required in the directory are:

- `collection.tsv`
- `qrels_train.txt`
- `queries_train.csv`
- `queries_test.csv`

## Usage

### Indexing
To index the documents using PyTerrier, run the following code:
```py
python baseline.py
```
This code will initialize PyTerrier, create an index of your dataset, and store it in the ./index directory.
### Scoring
The code for initializing the scorer is provided in the project. You can customize the scorer to meet your specific requirements by implementing the necessary functionality in the provided code section.