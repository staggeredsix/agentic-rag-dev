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
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from typing import Any, Dict, List, Tuple, Union
from urllib.parse import urlparse
import os
import shutil
import mimetypes

# Disable telemetry if the posthog library is available. Some vectorstore
# implementations use `posthog.capture()` with several arguments which can fail
# when telemetry is patched with a simple single-argument stub. To avoid noisy
# errors during document uploads, replace the capture function with a no-op that accepts
# arbitrary parameters.
try:  # pragma: no cover - best effort safeguard
    import posthog

    def _noop_capture(*_args, **_kwargs) -> None:
        """Ignore all telemetry events."""

    posthog.capture = _noop_capture  # type: ignore[attr-defined]
except Exception:
    pass

# Default model for local embeddings
EMBEDDINGS_MODEL = 'llama3-chatqa:8b'
# Base URL for the Ollama service. Defaults to the service name used in the
# docker compose network.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# Vector store configuration

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "/project/data/faiss_index")

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

        vectorstore = FAISS.from_documents(
            documents=doc_splits,
            embedding=OllamaEmbeddings(model=EMBEDDINGS_MODEL, base_url=OLLAMA_BASE_URL),
        )
        vectorstore.save_local(FAISS_INDEX_PATH)
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
    """Embed and store the split documents into a FAISS vectorstore."""
    try:
        print(f"[embed_documents] Embedding {len(doc_splits)} chunks using model: {EMBEDDINGS_MODEL}")

        vectorstore = FAISS.from_documents(
            doc_splits,
            OllamaEmbeddings(model=EMBEDDINGS_MODEL, base_url=OLLAMA_BASE_URL),
        )
        vectorstore.save_local(FAISS_INDEX_PATH)
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




def _clear():
    """Delete the persisted FAISS index, if it exists."""
    try:
        if os.path.isdir(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)
            print(f"[clear] Index at '{FAISS_INDEX_PATH}' cleared.")
    except Exception as e:
        print(f"[clear] Failed to clear vector store: {e}")


class _EmptyRetriever:
    def invoke(self, _query: str):
        return []


def get_retriever():
    """Return the retriever object of the FAISS vector store."""
    try:
        if os.path.isdir(FAISS_INDEX_PATH):
            vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH,
                embeddings=OllamaEmbeddings(
                    model=EMBEDDINGS_MODEL, base_url=OLLAMA_BASE_URL
                ),
            )
            return vectorstore.as_retriever()
    except Exception as e:
        print(f"[get_retriever] Failed to load index: {e}")

    print(f"[get_retriever] Returning empty retriever.")
    return _EmptyRetriever()
