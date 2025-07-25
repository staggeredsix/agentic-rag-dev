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

import os
import json
import re

from typing_extensions import TypedDict
from typing import List, Optional

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

from chatui.utils import database, nim


# Tavily related parameters and exceptions
DEFAULT_TAVILY_K = 3
TAVILY_K = int(os.getenv("TAVILY_K", DEFAULT_TAVILY_K)) 

class TavilyAPIError(Exception):
    """Raised when Tavily returns invalid or unauthorized results."""
    pass


def extract_json_from_text(text: str, expected_key: str) -> dict:
    """
    Extract JSON from text that may contain additional content or be incomplete.
    
    Args:
        text: The raw text output from the model
        expected_key: Either "datasource" for router or "score" for graders
    
    Returns:
        dict: Valid JSON object
    """
    # Clean the text
    text = text.strip()
    
    # Try to parse the text directly first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Handle incomplete JSON - try to reconstruct it
    if expected_key == "datasource":
        # Look for datasource values
        if "web_search" in text.lower():
            return {"datasource": "web_search"}
        elif "vectorstore" in text.lower():
            return {"datasource": "vectorstore"}
        # Try to extract from partial JSON
        if '"vectorstore"' in text or 'vectorstore' in text:
            return {"datasource": "vectorstore"}
        else:
            return {"datasource": "web_search"}  # Default fallback
            
    elif expected_key == "score":
        # Look for score values
        if "yes" in text.lower():
            return {"score": "yes"}
        elif "no" in text.lower():
            return {"score": "no"}
        else:
            return {"score": "no"}  # Default fallback for safety
    
    # Final fallback
    if expected_key == "datasource":
        return {"datasource": "web_search"}
    else:
        return {"score": "no"}


### State

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]
    generator_model_id: str
    router_model_id: str
    retrieval_model_id: str
    hallucination_model_id: str
    answer_model_id: str
    prompt_generator: str
    prompt_router: str
    prompt_retrieval: str
    prompt_relationship: str
    prompt_hallucination: str
    prompt_answer: str
    router_use_nim: bool
    retrieval_use_nim: bool
    generator_use_nim: bool
    hallucination_use_nim: bool
    answer_use_nim: bool
    nim_generator_ip: str
    nim_router_ip: str
    nim_retrieval_ip: str
    nim_hallucination_ip: str
    nim_answer_ip: str
    nim_generator_port: str
    nim_router_port: str
    nim_retrieval_port: str
    nim_hallucination_port: str
    nim_answer_port: str
    nim_generator_id: str
    nim_router_id: str
    nim_retrieval_id: str
    nim_hallucination_id: str
    nim_answer_id: str
    nim_generator_gpu_type: Optional[str]
    nim_generator_gpu_count: Optional[str]
    nim_router_gpu_type: Optional[str]
    nim_router_gpu_count: Optional[str]
    nim_retrieval_gpu_type: Optional[str]
    nim_retrieval_gpu_count: Optional[str]
    nim_hallucination_gpu_type: Optional[str]
    nim_hallucination_gpu_count: Optional[str]
    nim_answer_gpu_type: Optional[str]
    nim_answer_gpu_count: Optional[str]


from langchain.schema import Document

### Nodes


def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    retriever = database.get_retriever()
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    prompt = PromptTemplate(
        template=state["prompt_generator"],
        input_variables=["question", "document"],
    )

    llm = nim.CustomChatOpenAI(custom_endpoint=state["nim_generator_ip"],
                               port=state["nim_generator_port"] if len(state["nim_generator_port"]) > 0 else "11434",
                               model_name=state["nim_generator_id"] if len(state["nim_generator_id"]) > 0 else "llama3-chatqa:8b",
                               gpu_type=state.get("nim_generator_gpu_type"),
                               gpu_count=state.get("nim_generator_gpu_count"),
                               temperature=0.7)

    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    prompt = PromptTemplate(
        template=state["prompt_retrieval"],
        input_variables=["question", "document"],
    )

    llm = nim.CustomChatOpenAI(custom_endpoint=state["nim_retrieval_ip"],
                               port=state["nim_retrieval_port"] if len(state["nim_retrieval_port"]) > 0 else "11434",
                               model_name=state["nim_retrieval_id"] if len(state["nim_retrieval_id"]) > 0 else "llama3-chatqa:8b",
                               gpu_type=state.get("nim_retrieval_gpu_type"),
                               gpu_count=state.get("nim_retrieval_gpu_count"),
                               temperature=0.0)

    retrieval_grader = prompt | llm | StrOutputParser()
    for d in documents:
        try:
            raw_output = retrieval_grader.invoke({"question": question, "document": d.page_content})
            print(f"Raw retrieval output: {repr(raw_output)}")  # Debug output
            score = extract_json_from_text(raw_output, "score")
            grade = score["score"]
        except Exception as exc:
            print(f"error parsing retrieval grader output: {exc}")
            grade = "no"
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            continue
    # Assess the collection of filtered docs to ensure they collectively answer the question
    docs_content = "\n".join([d.page_content for d in filtered_docs])
    relationship_prompt = PromptTemplate(
        template=state["prompt_relationship"],
        input_variables=["question", "documents"],
    )
    relationship_grader = relationship_prompt | llm | StrOutputParser()
    try:
        raw_output = relationship_grader.invoke({"question": question, "documents": docs_content})
        print(f"Raw relationship output: {repr(raw_output)}")
        score = extract_json_from_text(raw_output, "score")
        docset_grade = score["score"]
    except Exception as exc:
        print(f"error parsing relationship grader output: {exc}")
        docset_grade = "no"

    web_search = "Yes" if len(filtered_docs) < 1 or docset_grade.lower() != "yes" else "No"
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    web_search_tool = TavilySearchResults(max_results=TAVILY_K)

    # Web search
    try:
        docs = web_search_tool.invoke({"query": question})

        # Manually validate Tavily returned what we expect
        if not isinstance(docs, list) or not all(isinstance(d, dict) and "content" in d for d in docs):
            raise TavilyAPIError(f"Invalid response from Tavily: {docs}")

        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)

        return {"documents": documents, "question": question}

    except Exception as e:
        raise TavilyAPIError(f"Tavily web search failed: {e}")


### Conditional edge


def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    prompt = PromptTemplate(
        template=state["prompt_router"],
        input_variables=["question"],
    )

    llm = nim.CustomChatOpenAI(custom_endpoint=state["nim_router_ip"],
                               port=state["nim_router_port"] if len(state["nim_router_port"]) > 0 else "11434",
                               model_name=state["nim_router_id"] if len(state["nim_router_id"]) > 0 else "llama3-chatqa:8b",
                               gpu_type=state.get("nim_router_gpu_type"),
                               gpu_count=state.get("nim_router_gpu_count"),
                               temperature=0.0)

    question_router = prompt | llm | StrOutputParser()
    try:
        raw_output = question_router.invoke({"question": question})
        print(f"Raw router output: {repr(raw_output)}")  # Debug output
        source = extract_json_from_text(raw_output, "datasource")
    except Exception as exc:
        print(f"error parsing router output: {exc}")
        source = {"datasource": "web_search"}
    print(source)
    if source["datasource"] == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


### Conditional edge


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    prompt = PromptTemplate(
        template=state["prompt_hallucination"],
        input_variables=["generation", "documents"],
    )

    llm = nim.CustomChatOpenAI(custom_endpoint=state["nim_hallucination_ip"],
                               port=state["nim_hallucination_port"] if len(state["nim_hallucination_port"]) > 0 else "11434",
                               model_name=state["nim_hallucination_id"] if len(state["nim_hallucination_id"]) > 0 else "llama3-chatqa:8b",
                               gpu_type=state.get("nim_hallucination_gpu_type"),
                               gpu_count=state.get("nim_hallucination_gpu_count"),
                               temperature=0.0)

    hallucination_grader = prompt | llm | StrOutputParser()

    try:
        raw_output = hallucination_grader.invoke({"documents": documents, "generation": generation})
        print(f"Raw hallucination output: {repr(raw_output)}")  # Debug output
        score = extract_json_from_text(raw_output, "score")
        grade = score["score"]
    except Exception as exc:
        print(f"error parsing hallucination grader output: {exc}")
        grade = "no"

    # Check hallucination
    prompt = PromptTemplate(
        template=state["prompt_answer"],
        input_variables=["generation", "question"],
    )

    llm = nim.CustomChatOpenAI(custom_endpoint=state["nim_answer_ip"],
                               port=state["nim_answer_port"] if len(state["nim_answer_port"]) > 0 else "11434",
                               model_name=state["nim_answer_id"] if len(state["nim_answer_id"]) > 0 else "llama3-chatqa:8b",
                               gpu_type=state.get("nim_answer_gpu_type"),
                               gpu_count=state.get("nim_answer_gpu_count"),
                               temperature=0.0)

    answer_grader = prompt | llm | StrOutputParser()
    
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        try:
            raw_output = answer_grader.invoke({"question": question, "generation": generation})
            print(f"Raw answer output: {repr(raw_output)}")  # Debug output
            score = extract_json_from_text(raw_output, "score")
            grade = score["score"]
        except Exception as exc:
            print(f"error parsing answer grader output: {exc}")
            grade = "no"
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"