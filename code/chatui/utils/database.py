# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader,
    UnstructuredPDFLoader,
    TextLoader,
    CSVLoader,
)
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings

from typing import Any, Dict, List, Tuple, Union
from urllib.parse import urlparse
import os
import shutil
import mimetypes

# Default model for local embeddings
EMBEDDINGS_MODEL = 'llama3:8b-instruct'
# Base URL for the Ollama service. Defaults to the service name used in the
# docker compose network.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# Milvus connection configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")  # Use service name from compose.yaml
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# Custom Milvus connection args
CUSTOM_MILVUS_CONNECTION = {
    "host": MILVUS_HOST,
    "port": int(MILVUS_PORT),
}

# Milvus collection configuration
import re

DEFAULT_COLLECTION_NAME = "rag_milvus"
RAW_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", DEFAULT_COLLECTION_NAME)
COLLECTION_NAME = re.sub(r"[^0-9a-zA-Z_]", "_", RAW_COLLECTION_NAME)

# Set the chunk size and overlap for the text splitter. Uses defaults but allows them to be set as environment variables.
DEFAULT_CHUNK_SIZE = 250
DEFAULT_CHUNK_OVERLAP = 0

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))


# Adding nltk data
import nltk

def download_nltk_if_missing():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

download_nltk_if_missing()

    
# Functions for dealing with URLs
def is_valid_url(url: str) -> bool:
    """ This is a helper function for checking if the URL is valid. It isn't fail proof, but it will catch most common errors. """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def safe_load(url):
    """ This is a helper function for loading the URL. It protects against false negatives from is_value_url and 
        filters for actual web pages. Returns None if it fails.
    """
    try:
        return WebBaseLoader(url).load()
    except Exception as e:
        print(f"[upload] Skipping {url}: {e}")
        return None


def upload(urls: List[str]):
    """ This is a helper function for parsing the user inputted URLs and uploading them into the vector store. """

    urls = [url for url in urls if is_valid_url(url)]

    docs = []
    for url in urls:
        result = safe_load(url)
        if result is not None:
            docs.append(result)

    docs_list = [item for sublist in docs for item in sublist]

    if not docs_list:
        # If no documents were loaded, return None
        print("[upload] No URLs provided.")
        return None
    
    try:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Use Milvus with custom connection args
        vectorstore = Milvus.from_documents(
            documents=doc_splits,
            embedding=OllamaEmbeddings(model=EMBEDDINGS_MODEL, base_url=OLLAMA_BASE_URL),

            collection_name=COLLECTION_NAME,

            connection_args=CUSTOM_MILVUS_CONNECTION,
            drop_old=True,
        )
        return vectorstore

    except Exception as e:
        print(f"[upload] Vectorstore creation failed: {e}")
        return None


# Functions for dealing with file uploads/embeddings

## Helper functions

def load_documents_from_files(file_paths: List[str]) -> List[Any]:
    """Load and return documents from supported file types."""
    docs = []

    for fpath in file_paths:
        ext = os.path.splitext(fpath)[-1].lower()

        loader_cls = {
            ".pdf": UnstructuredPDFLoader,
            ".txt": TextLoader,
            ".md": TextLoader,
            ".csv": CSVLoader
        }.get(ext)

        if loader_cls is None:
            print(f"[load_documents] Skipping unsupported file type: {fpath}")
            continue

        try:
            loaded = loader_cls(fpath).load()
            docs.append(loaded)
        except Exception as e:
            print(f"[load_documents] Failed to load {fpath}: {e}")

    return [item for sublist in docs for item in sublist]


def split_documents(docs: List[Any]):
    """Split documents into smaller chunks using recursive splitter."""
    print(f"[split_documents] Splitting {len(docs)} docs with chunk size {CHUNK_SIZE}, overlap {CHUNK_OVERLAP}")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)

def embed_documents(doc_splits: List[Any]):
    """Embed and store the split documents into Milvus vectorstore."""
    try:
        print(f"[embed_documents] Embedding {len(doc_splits)} chunks using model: {EMBEDDINGS_MODEL}")

        vectorstore = Milvus.from_documents(
            doc_splits,
            OllamaEmbeddings(model=EMBEDDINGS_MODEL, base_url=OLLAMA_BASE_URL),

            collection_name=COLLECTION_NAME,

            connection_args=CUSTOM_MILVUS_CONNECTION,
            drop_old=True,
        )
        return vectorstore

    except Exception as e:
        print(f"[embed_documents] Vectorstore creation failed: {e}")
        return None


## Main function that use helper functions 

def upload_files(file_paths: List[str]):
    """Upload files into the vector store pipeline."""

    if not file_paths:
        print("[upload_files] No documents provided.")
        return None

    try:
        docs_list = load_documents_from_files(file_paths)

        if not docs_list:
            print("[upload_files] No documents successfully loaded.")
            return None

        doc_splits = split_documents(docs_list)
        return embed_documents(doc_splits)

    except Exception as e:
        print(f"[upload_files] Pipeline failed: {e}")
        return None




def _clear(

    collection_name: str = COLLECTION_NAME,

):
    """Clear the Milvus collection by dropping and recreating it."""
    try:
        Milvus(
            embedding_function=OllamaEmbeddings(model=EMBEDDINGS_MODEL, base_url=OLLAMA_BASE_URL),
            collection_name=collection_name,
            connection_args=CUSTOM_MILVUS_CONNECTION,
            drop_old=True,
        )
        print(f"[clear] Collection '{collection_name}' cleared.")
    except Exception as e:
        print(f"[clear] Failed to clear vector store: {e}")


def get_retriever():
    """Return the retriever object of the Milvus vector store."""
    vectorstore = Milvus(
        embedding_function=OllamaEmbeddings(model=EMBEDDINGS_MODEL, base_url=OLLAMA_BASE_URL),

        collection_name=COLLECTION_NAME,
        connection_args=CUSTOM_MILVUS_CONNECTION,
    )
    retriever = vectorstore.as_retriever()
    return retriever
