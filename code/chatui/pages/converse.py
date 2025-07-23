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

### This module contains the chatui gui for having a conversation. ###

import functools
from typing import Any, Dict, List, Tuple, Union

import gradio as gr
import shutil
import os
import subprocess
import time
import sys
import json

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError 

from requests.exceptions import HTTPError
import traceback


from chatui.utils.error_messages import QUERY_ERROR_MESSAGES
from chatui.utils.graph import TavilyAPIError

# UI names and labels
SELF_HOSTED_TAB_NAME = "Self-Hosted Endpoint"
HOST_NAME = "Local NIM or Remote IP/Hostname"
HOST_PORT = "Host Port"
HOST_MODEL = "Model Name"


# Set recursion limit 
DEFAULT_RECURSION_LIMIT = 10
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", DEFAULT_RECURSION_LIMIT))


# Default model configurations for self-hosted only setup
DEFAULT_ROUTER_MODEL = 'phi4-reasoning:14b'
DEFAULT_RETRIEVAL_MODEL = 'phi4-reasoning:14b'  
DEFAULT_GENERATOR_MODEL = 'phi4-reasoning:14b'
DEFAULT_HALLUCINATION_MODEL = 'llama3-chatqa:8b'
DEFAULT_ANSWER_MODEL = 'llama3-chatqa:8b'

# URLs for default example docs for the RAG.
doc_links = (
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/overview/introduction.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/installation/overview.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/installation/windows.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/installation/macos.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/installation/ubuntu-local.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/installation/ubuntu-remote.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/quickstart/quickstart-basic.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/quickstart/quickstart-cli.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/quickstart/quickstart-environment.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/quickstart/quickstart-environment-cli.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/quickstart/quickstart-custom-app.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/quickstart/quickstart-hybrid-rag.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/quickstart/example-projects.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/locations/remote.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/projects/base-environments.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/components.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/cli.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/troubleshooting/troubleshooting.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/troubleshooting/logging.html",
    "https://raw.githubusercontent.com/NVIDIA/workbench-example-agentic-rag/refs/heads/main/README.md",
)
EXAMPLE_LINKS_LEN = 10

EXAMPLE_LINKS = "\n".join(doc_links)

from chatui import assets, chat_client
from chatui.prompts import prompts_llama3, prompts_mistral
from chatui.utils import compile, database, logger

from langgraph.graph import END, StateGraph

PATH = "/"
TITLE = "Agentic RAG: Chat UI"
OUTPUT_TOKENS = 250
MAX_DOCS = 5

### Load in CSS here for components that need custom styling. ###

_LOCAL_CSS = """
#contextbox {
    overflow-y: scroll !important;
    max-height: 400px;
}

#params .tabs {
    display: flex;
    flex-direction: column;
    flex-grow: 1;
}
#params .tabitem[style="display: block;"] {
    flex-grow: 1;
    display: flex !important;
}
#params .gap {
    flex-grow: 1;
}
#params .form {
    flex-grow: 1 !important;
}
#params .form > :last-child{
    flex-grow: 1;
}
#accordion {
}
#rag-inputs .svelte-1gfkn6j .svelte-s1r2yt .svelte-cmf5ev {
    color: #76b900 !important;
}
.mode-banner {
    font-size: 1.05rem;
    font-weight: 500;
    background-color: #f0f4f8;
    padding: 0.5em 0.75em;
    border-left: 2px solid #76b900;
    margin-bottom: 0.5em;
    border-radius: 2px;
}
"""

sys.stdout = logger.Logger("/project/code/output.log")

def build_page(client: chat_client.ChatClient) -> gr.Blocks:
    """
    Build the gradio page to be mounted in the frame.
    
    Parameters: 
        client (chat_client.ChatClient): The chat client running the application. 
    
    Returns:
        page (gr.Blocks): A Gradio page.
    """
    kui_theme, kui_styles = assets.load_theme("kaizen")
    
    """ Compile the agentic graph. """
    
    workflow = compile.compile_graph()
    app = workflow.compile()

    with gr.Blocks(title=TITLE, theme=kui_theme, css=kui_styles + _LOCAL_CSS) as page:
        gr.Markdown(f"# {TITLE}")

        """ Build the Chat Application. """
        
        with gr.Row(equal_height=True):

            # Left Column will display the chatbot
            with gr.Column(scale=16, min_width=350):

                # Main chatbot panel. 
                with gr.Row(equal_height=True):
                    with gr.Column(min_width=350):
                        chatbot = gr.Chatbot(show_label=False, height=500)

                # Message box for user input
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3, min_width=450):
                        msg = gr.Textbox(
                            show_label=False,
                            placeholder="Enter text and press ENTER",
                            container=False,
                            interactive=True,
                        )

                    with gr.Column(scale=1, min_width=150):
                        _ = gr.ClearButton([msg, chatbot], value="Clear Chat History")

                # Sample questions that users can click on to use
                with gr.Row(equal_height=True):
                    sample_query_1 = gr.Button("How do I add an integration in the CLI?", variant="secondary", size="sm", interactive=True)
                    sample_query_2 = gr.Button("How do I fix an inaccessible remote Location?", variant="secondary", size="sm", interactive=True)
                
                with gr.Row(equal_height=True):
                    sample_query_3 = gr.Button("What are the NVIDIA-provided default base environments?", variant="secondary", size="sm", interactive=True)
                    sample_query_4 = gr.Button("How do I create a support bundle for troubleshooting?", variant="secondary", size="sm", interactive=True)
            
            # Hidden column to be rendered when the user collapses all settings.
            with gr.Column(scale=1, min_width=100, visible=False) as hidden_settings_column:
                show_settings = gr.Button(value="< Expand", size="sm")
            
            # Right column to display all relevant settings
            with gr.Column(scale=12, min_width=350) as settings_column:
                with gr.Tabs(selected=0) as settings_tabs:

                    with gr.TabItem("Quickstart", id=0) as instructions_tab:
                        
                        # Diagram of the agentic websearch RAG workflow
                        with gr.Row():
                            agentic_flow = gr.Image("/project/code/chatui/static/agentic-flow.png", 
                                                    show_label=False,
                                                    show_download_button=False,
                                                    interactive=False)

                        with gr.Column():
                            step_1_btn = gr.Button("Step 1: Submit a sample query", elem_id="rag-inputs")
                            step_1 = gr.Markdown(
                                """
                                ### Purpose: Generate and evaluate a generic response&nbsp;<ins>without</ins>&nbsp;RAG

                                * Ensure ``TAVILY_API_KEY`` is configured in AI Workbench.
                                * Select a sample query from the chatbot on the left-hand side of the window.
                                * Wait for the response to generate and evaluate the relevance of the response.
                                """,
                                visible=True
                            )

                            step_2_btn = gr.Button("Step 2: Upload the sample dataset", elem_id="rag-inputs")
                            step_2 = gr.Markdown(
                                """
                                ### Purpose: Populate the RAG database with useful context to augment responses

                                * Select the **Documents** tab on the right-hand side of the browser window.
                                * Select **Add to Context** under the sample webpage dataset.
                                * Wait for the upload to complete.
                                """,
                                visible=False
                            )

                            step_3_btn = gr.Button("Step 3: Resubmit the sample query", elem_id="rag-inputs")
                            step_3 = gr.Markdown(
                                """
                                ### Purpose: Generate and evaluate a generic response&nbsp;<ins>with</ins>&nbsp;added RAG context

                                * Select the same sample query from Step 1.
                                * Wait for the response to generate and evaluate the relevance of the response.
                                """,
                                visible=False
                            )

                            step_4_btn = gr.Button("Step 4: Monitor the results", elem_id="rag-inputs")
                            step_4 = gr.Markdown(
                                """
                                ### Purpose: Understand the actions the agent takes in generating responses

                                * Select the **Monitor** tab on the right-hand side of the browser window.
                                * Take a look at the actions taken by the agent under **Actions Console**.
                                * Take a look at the latest response generation details under **Response Trace**.
                                """,
                                visible=False
                            )

                            step_5_btn = gr.Button("Step 5: Next steps", elem_id="rag-inputs")
                            step_5 = gr.Markdown(
                                """
                                ### Purpose: Customize the project to your own documents and datasets

                                * To customize, clear out the database and upload your own data under **Documents**.
                                * Configure the **Router Prompt** to your RAG topic(s) under the **Models** tab.
                                * Submit a custom query to the RAG agent and evaluate the response.
                                """,
                                visible=False
                            )


                    # Settings for each component model of the agentic workflow
                    with gr.TabItem("Models", id=1) as agent_settings:
                            gr.Markdown(
                                        """
                                        ##### Use the Models tab to configure individual model components
                                        - All models are configured for self-hosted Ollama endpoints
                                        - Configure the IP/hostname and port for each component
                                        - Default models: Router/Retrieval/Generator use phi4-reasoning:14b, Hallucination/Answer use llama3-chatqa:8b
                                        - (optional) Customize component behavior by changing the prompts
                                        """
                            )
                            gr.HTML('<hr style="border:1px solid #ccc; margin: 10px 0;">')
                                    
                            ########################
                            ##### ROUTER MODEL #####
                            ########################
                            router_btn = gr.Button("Router", variant="sm")
                            with gr.Group(visible=False) as group_router:
                                        
                                with gr.Row():
                                    nim_router_ip = gr.Textbox(
                                        value = "ollama",
                                        label=HOST_NAME,
                                        info="Local microservice OR IP address running a remote microservice",
                                        elem_id="rag-inputs",
                                        scale=2
                                    )
                                    nim_router_port = gr.Textbox(
                                        placeholder="11434",
                                        label=HOST_PORT,
                                        info="Optional, (default: 11434)",
                                        elem_id="rag-inputs",
                                        scale=1
                                    )
                                nim_router_id = gr.Textbox(
                                    value = DEFAULT_ROUTER_MODEL,
                                    label=HOST_MODEL,
                                    info=f"Default: {DEFAULT_ROUTER_MODEL}",
                                    elem_id="rag-inputs",
                                    interactive=True
                                )

                                with gr.Accordion("Configure the Router Prompt", 
                                                elem_id="rag-inputs", open=False) as accordion_router:
                                    prompt_router = gr.Textbox(value=prompts_llama3.router_prompt,
                                                            lines=12,
                                                            show_label=False,
                                                            interactive=True)
        
                            ##################################
                            ##### RETRIEVAL GRADER MODEL #####
                            ##################################
                            retrieval_btn = gr.Button("Retrieval Grader", variant="sm")
                            with gr.Group(visible=False) as group_retrieval:
                                        
                                with gr.Row():
                                    nim_retrieval_ip = gr.Textbox(
                                        value = "ollama",
                                        label=HOST_NAME,
                                        info="Local microservice OR IP address running a remote microservice",
                                        elem_id="rag-inputs",
                                        scale=2
                                    )
                                    nim_retrieval_port = gr.Textbox(
                                        placeholder="11434",
                                        label=HOST_PORT,
                                        info="Optional, (default: 11434)",
                                        elem_id="rag-inputs",
                                        scale=1
                                    )
                                nim_retrieval_id = gr.Textbox(
                                    value = DEFAULT_RETRIEVAL_MODEL,
                                    label=HOST_MODEL,
                                    info=f"Default: {DEFAULT_RETRIEVAL_MODEL}",
                                    elem_id="rag-inputs",
                                    interactive=True
                                )                                        
                                
                                with gr.Accordion("Configure the Retrieval Grader Prompt", 
                                                elem_id="rag-inputs", open=False) as accordion_retrieval:
                                    prompt_retrieval = gr.Textbox(value=prompts_llama3.retrieval_prompt,
                                                                        lines=21,
                                                                        show_label=False,
                                                                        interactive=True)
        
                            ###########################
                            ##### GENERATOR MODEL #####
                            ###########################
                            generator_btn = gr.Button("Generator", variant="sm")
                            with gr.Group(visible=False) as group_generator:
                                        
                                with gr.Row():
                                    nim_generator_ip = gr.Textbox(
                                        value = "ollama",
                                        label=HOST_NAME,
                                        info="Local microservice OR IP address running a remote microservice",
                                        elem_id="rag-inputs",
                                        scale=2
                                    )
                                    nim_generator_port = gr.Textbox(
                                        placeholder="11434",
                                        label=HOST_PORT,
                                        info="Optional, (default: 11434)",
                                        elem_id="rag-inputs",
                                        scale=1
                                    )
                                nim_generator_id = gr.Textbox(
                                    value = DEFAULT_GENERATOR_MODEL,
                                    label=HOST_MODEL,
                                    info=f"Default: {DEFAULT_GENERATOR_MODEL}",
                                    elem_id="rag-inputs",
                                    interactive=True
                                )
                                
                                with gr.Accordion("Configure the Generator Prompt", 
                                                elem_id="rag-inputs", open=False) as accordion_generator:
                                    prompt_generator = gr.Textbox(value=prompts_llama3.generator_prompt,
                                                            lines=15,
                                                            show_label=False,
                                                            interactive=True)
        
                            ######################################
                            ##### HALLUCINATION GRADER MODEL #####
                            ######################################
                            hallucination_btn = gr.Button("Hallucination Grader", variant="sm")
                            with gr.Group(visible=False) as group_hallucination:
                                        
                                with gr.Row():
                                    nim_hallucination_ip = gr.Textbox(
                                        value = "ollama",
                                        label=HOST_NAME,
                                        info="Local microservice OR IP address running a remote microservice",
                                        elem_id="rag-inputs",
                                        scale=2
                                    )
                                    nim_hallucination_port = gr.Textbox(
                                        placeholder="11434",
                                        label=HOST_PORT,
                                        info="Optional, (default: 11434)",
                                        elem_id="rag-inputs",
                                        scale=1
                                    )
                                nim_hallucination_id = gr.Textbox(
                                    value = DEFAULT_HALLUCINATION_MODEL,
                                    label=HOST_MODEL,
                                    info=f"Default: {DEFAULT_HALLUCINATION_MODEL}",
                                    elem_id="rag-inputs",
                                    interactive=True
                                )

                                with gr.Accordion("Configure the Hallucination Prompt", 
                                                elem_id="rag-inputs", open=False) as accordion_hallucination:
                                    prompt_hallucination = gr.Textbox(value=prompts_llama3.hallucination_prompt,
                                                                            lines=17,
                                                                            show_label=False,
                                                                            interactive=True)
        
                            ###############################
                            ##### ANSWER GRADER MODEL #####
                            ###############################
                            answer_btn = gr.Button("Answer Grader", variant="sm")
                            with gr.Group(visible=False) as group_answer:
                                        
                                with gr.Row():
                                    nim_answer_ip = gr.Textbox(
                                        value = "ollama",
                                        label=HOST_NAME,
                                        info="Local microservice OR IP address running a remote microservice",
                                        elem_id="rag-inputs",
                                        scale=2
                                    )
                                    nim_answer_port = gr.Textbox(
                                        placeholder="11434",
                                        label=HOST_PORT,
                                        info="Optional, (default: 11434)",
                                        elem_id="rag-inputs",
                                        scale=1
                                    )
                                nim_answer_id = gr.Textbox(
                                    value = DEFAULT_ANSWER_MODEL,
                                    label=HOST_MODEL,
                                    info=f"Default: {DEFAULT_ANSWER_MODEL}",
                                    elem_id="rag-inputs",
                                    interactive=True
                                    )   
                                        
                                with gr.Accordion("Configure the Answer Prompt", 
                                                elem_id="rag-inputs", open=False) as accordion_answer:
                                    prompt_answer = gr.Textbox(value=prompts_llama3.answer_prompt,
                                                                    lines=17,
                                                                    show_label=False,
                                                                    interactive=True)
                        
                    # Third tab item is for uploading to and clearing the vector database
                    with gr.TabItem("Documents", id=2) as document_settings:
                        gr.Markdown(
                            """                            
                            ##### Use the Documents tab to create a RAG context
                            - Webpages: Enter URLs of webpages for the context
                            - Files: Use files (pdf, csv, .txt) for the context
                            - Add to Context: Add documents to the context. Context is stored until you clear it.
                            - Clear Context: Resets the context to empty
                            """
                            )
                        gr.HTML('<hr style="border:1px solid #ccc; margin: 10px 0;">')
                        with gr.Tabs(selected=0) as document_tabs:
                            with gr.TabItem("Webpages", id=0) as url_tab:
                                url_docs = gr.Textbox(value=EXAMPLE_LINKS,
                                                      lines=EXAMPLE_LINKS_LEN, 
                                                      info="Enter a list of URLs, one per line", 
                                                      show_label=False, 
                                                      interactive=True)
                            
                                with gr.Row():
                                    url_docs_upload = gr.Button(value="Add to Context")
                                    url_docs_clear = gr.Button(value="Clear Context")

                            with gr.TabItem("Files", id=1) as pdf_tab:
                                docs_upload = gr.File(interactive=True, 
                                                          show_label=False, 
                                                          file_types=[".pdf", ".txt", ".csv", ".md"], 
                                                          file_count="multiple")
                                docs_clear = gr.Button(value="Clear Context")
    
                    # Fourth tab item is for the actions output console. 
                    with gr.TabItem("Monitor", id=3) as console_settings:
                        gr.Markdown(
                            """
                            ##### Use the Monitor tab to see the agent in action
                            - Actions Console: View the actions taken by the agent
                            - Response Trace: Full analysis behind the latest response
                            """
                            )
                        gr.HTML('<hr style="border:1px solid #ccc; margin: 10px 0;">')
                        with gr.Tabs(selected=0) as console_tabs:
                            with gr.TabItem("Actions Console", id=0) as actions_tab:
                                logs = gr.Textbox(show_label=False, lines=18, max_lines=18, interactive=False)
                            with gr.TabItem("Response Trace", id=1) as trace_tab:
                                actions = gr.JSON(
                                    scale=1,
                                    show_label=False,
                                    visible=True,
                                    elem_id="contextbox",
                                )
                    
                    # Fifth tab item is for collapsing the entire settings pane for readability. 
                    with gr.TabItem("Hide All Settings", id=4) as hide_all_settings:
                        gr.Markdown("")

        page.load(logger.read_logs, None, logs, every=1)

        """ These helper functions hide all other quickstart steps when one step is expanded. """

        def _toggle_quickstart_steps(step):
            steps = ["Step 1: Submit a sample query",
                     "Step 2: Upload the sample dataset",
                     "Step 3: Resubmit the sample query",
                     "Step 4: Monitor the results",
                     "Step 5: Next steps"]
            visible = [False, False, False, False, False]
            visible[steps.index(step)] = True
            return {
                step_1: gr.update(visible=visible[0]),
                step_2: gr.update(visible=visible[1]),
                step_3: gr.update(visible=visible[2]),
                step_4: gr.update(visible=visible[3]),
                step_5: gr.update(visible=visible[4]),
            }

        step_1_btn.click(_toggle_quickstart_steps, [step_1_btn], [step_1, step_2, step_3, step_4, step_5])
        step_2_btn.click(_toggle_quickstart_steps, [step_2_btn], [step_1, step_2, step_3, step_4, step_5])
        step_3_btn.click(_toggle_quickstart_steps, [step_3_btn], [step_1, step_2, step_3, step_4, step_5])
        step_4_btn.click(_toggle_quickstart_steps, [step_4_btn], [step_1, step_2, step_3, step_4, step_5])
        step_5_btn.click(_toggle_quickstart_steps, [step_5_btn], [step_1, step_2, step_3, step_4, step_5])

        """ These helper functions hide all settings when collapsed, and displays all settings when expanded. """

        def _toggle_hide_all_settings():
            return {
                settings_column: gr.update(visible=False),
                hidden_settings_column: gr.update(visible=True),
            }

        def _toggle_show_all_settings():
            return {
                settings_column: gr.update(visible=True),
                settings_tabs: gr.update(selected=0),
                hidden_settings_column: gr.update(visible=False),
            }

        hide_all_settings.select(_toggle_hide_all_settings, None, [settings_column, hidden_settings_column])
        show_settings.click(_toggle_show_all_settings, None, [settings_column, settings_tabs, hidden_settings_column])

        """ These helper functions hide the expanded component model settings when the Hide tab is clicked. """
        
        def _toggle_hide_router():
            return {
                group_router: gr.update(visible=False),
                router_btn: gr.update(visible=True),
            }

        def _toggle_hide_retrieval():
            return {
                group_retrieval: gr.update(visible=False),
                retrieval_btn: gr.update(visible=True),
            }

        def _toggle_hide_generator():
            return {
                group_generator: gr.update(visible=False),
                generator_btn: gr.update(visible=True),
            }

        def _toggle_hide_hallucination():
            return {
                group_hallucination: gr.update(visible=False),
                hallucination_btn: gr.update(visible=True),
            }

        def _toggle_hide_answer():
            return {
                group_answer: gr.update(visible=False),
                answer_btn: gr.update(visible=True),
            }

        """ These helper functions upload and clear the documents and webpages to/from the ChromaDB. """

        def _upload_documents_files(files, progress=gr.Progress()):
            progress(0.25, desc="Initializing Task")
            time.sleep(0.75)
            progress(0.5, desc="Uploading Docs")
            database.upload_files(files)
            progress(0.75, desc="Cleaning Up")
            time.sleep(0.75)
            return {
                url_docs_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                docs_clear: gr.update(value="Clear Docs", variant="secondary", interactive=True),
                agentic_flow: gr.update(visible=True),
            }

        def _upload_documents(docs: str, progress=gr.Progress()):
            progress(0.2, desc="Initializing Task")
            time.sleep(0.75)
            progress(0.4, desc="Processing URL List")
            docs_list = docs.splitlines()
            progress(0.6, desc="Creating Context")
            vectorstore = database.upload(docs_list)
            progress(0.8, desc="Cleaning Up")
            time.sleep(0.75)
            if vectorstore is None:
                return {
                    url_docs_upload: gr.update(value="No valid URLS - Try again", variant="secondary", interactive=True),
                    url_docs_clear: gr.update(value="Clear Context", variant="secondary", interactive=False),
                    docs_clear: gr.update(value="Clear Context", variant="secondary", interactive=False),
                    agentic_flow: gr.update(visible=False),  # or leave as-is if flow is independent
                }
            return {
                url_docs_upload: gr.update(value="Context Created", variant="primary", interactive=False),
                url_docs_clear: gr.update(value="Clear Context", variant="secondary", interactive=True),
                docs_clear: gr.update(value="Clear Context", variant="secondary", interactive=True),
                agentic_flow: gr.update(visible=True),
            }

        def _clear_documents(progress=gr.Progress()):
            progress(0.25, desc="Initializing Task")
            time.sleep(0.75)
            progress(0.5, desc="Clearing Context")
            database._clear()
            progress(0.75, desc="Cleaning Up")
            time.sleep(0.75)
            return {
                url_docs_upload: gr.update(value="Add to Context", variant="secondary", interactive=True),
                url_docs_clear: gr.update(value="Context Cleared", variant="primary", interactive=False),
                docs_upload: gr.update(value=None),
                docs_clear: gr.update(value="Context Cleared", variant="primary", interactive=False),
                agentic_flow: gr.update(visible=True),
            }

        url_docs_upload.click(_upload_documents, [url_docs], [url_docs_upload, url_docs_clear, docs_clear, agentic_flow])
        url_docs_clear.click(_clear_documents, [], [url_docs_upload, url_docs_clear, docs_upload, docs_clear, agentic_flow])
        docs_upload.upload(_upload_documents_files, [docs_upload], [url_docs_clear, docs_clear, agentic_flow])
        docs_clear.click(_clear_documents, [], [url_docs_upload, url_docs_clear, docs_upload, docs_clear, agentic_flow])

        """ These helper functions set state and prompts when either the NIM or API Endpoint tabs are selected. """
        
        def _toggle_model_tab():
            return {
                group_router: gr.update(visible=False),
                group_retrieval: gr.update(visible=False),
                group_generator: gr.update(visible=False),
                group_hallucination: gr.update(visible=False),
                group_answer: gr.update(visible=False),
                router_btn: gr.update(visible=True),
                retrieval_btn: gr.update(visible=True),
                generator_btn: gr.update(visible=True),
                hallucination_btn: gr.update(visible=True),
                answer_btn: gr.update(visible=True),
            }
        
        agent_settings.select(_toggle_model_tab, [], [group_router,
                                                      group_retrieval,
                                                      group_generator,
                                                      group_hallucination,
                                                      group_answer,
                                                      router_btn,
                                                      retrieval_btn,
                                                      generator_btn,
                                                      hallucination_btn,
                                                      answer_btn])

        """ This helper function ensures only one component model settings are expanded at a time when selected. """

        def _toggle_model(btn: str):
            if btn == "Router":
                group_visible = [True, False, False, False, False]
                button_visible = [False, True, True, True, True]
            elif btn == "Retrieval Grader":
                group_visible = [False, True, False, False, False]
                button_visible = [True, False, True, True, True]
            elif btn == "Generator":
                group_visible = [False, False, True, False, False]
                button_visible = [True, True, False, True, True]
            elif btn == "Hallucination Grader":
                group_visible = [False, False, False, True, False]
                button_visible = [True, True, True, False, True]
            elif btn == "Answer Grader":
                group_visible = [False, False, False, False, True]
                button_visible = [True, True, True, True, False]
            return {
                group_router: gr.update(visible=group_visible[0]),
                group_retrieval: gr.update(visible=group_visible[1]),
                group_generator: gr.update(visible=group_visible[2]),
                group_hallucination: gr.update(visible=group_visible[3]),
                group_answer: gr.update(visible=group_visible[4]),
                router_btn: gr.update(visible=button_visible[0]),
                retrieval_btn: gr.update(visible=button_visible[1]),
                generator_btn: gr.update(visible=button_visible[2]),
                hallucination_btn: gr.update(visible=button_visible[3]),
                answer_btn: gr.update(visible=button_visible[4]),
            }

        router_btn.click(_toggle_model, [router_btn], [group_router,
                                                       group_retrieval,
                                                       group_generator,
                                                       group_hallucination,
                                                       group_answer,
                                                       router_btn,
                                                       retrieval_btn,
                                                       generator_btn,
                                                       hallucination_btn,
                                                       answer_btn])
        
        retrieval_btn.click(_toggle_model, [retrieval_btn], [group_router,
                                                                           group_retrieval,
                                                                           group_generator,
                                                                           group_hallucination,
                                                                           group_answer,
                                                                           router_btn,
                                                                           retrieval_btn,
                                                                           generator_btn,
                                                                           hallucination_btn,
                                                                           answer_btn])
        
        generator_btn.click(_toggle_model, [generator_btn], [group_router,
                                                             group_retrieval,
                                                             group_generator,
                                                             group_hallucination,
                                                             group_answer,
                                                             router_btn,
                                                             retrieval_btn,
                                                             generator_btn,
                                                             hallucination_btn,
                                                             answer_btn])
        
        hallucination_btn.click(_toggle_model, [hallucination_btn], [group_router,
                                                                                   group_retrieval,
                                                                                   group_generator,
                                                                                   group_hallucination,
                                                                                   group_answer,
                                                                                   router_btn,
                                                                                   retrieval_btn,
                                                                                   generator_btn,
                                                                                   hallucination_btn,
                                                                                   answer_btn])
        
        answer_btn.click(_toggle_model, [answer_btn], [group_router,
                                                                     group_retrieval,
                                                                     group_generator,
                                                                     group_hallucination,
                                                                     group_answer,
                                                                     router_btn,
                                                                     retrieval_btn,
                                                                     generator_btn,
                                                                     hallucination_btn,
                                                                     answer_btn])

        """ This helper function builds out the submission function call when a user submits a query. """
        
        _my_build_stream = functools.partial(_stream_predict, client, app)

        # Submit a sample query
        sample_query_1.click(
            _my_build_stream, [sample_query_1, 
                               prompt_generator,
                               prompt_router,
                               prompt_retrieval,
                               prompt_hallucination,
                               prompt_answer,
                               nim_generator_ip,
                               nim_router_ip,
                               nim_retrieval_ip,
                               nim_hallucination_ip,
                               nim_answer_ip,
                               nim_generator_port,
                               nim_router_port,
                               nim_retrieval_port,
                               nim_hallucination_port,
                               nim_answer_port,
                               nim_generator_id,
                               nim_router_id,
                               nim_retrieval_id,
                               nim_hallucination_id,
                               nim_answer_id,
                               chatbot], [msg, chatbot, actions]
        )

        sample_query_2.click(
            _my_build_stream, [sample_query_2, 
                               prompt_generator,
                               prompt_router,
                               prompt_retrieval,
                               prompt_hallucination,
                               prompt_answer,
                               nim_generator_ip,
                               nim_router_ip,
                               nim_retrieval_ip,
                               nim_hallucination_ip,
                               nim_answer_ip,
                               nim_generator_port,
                               nim_router_port,
                               nim_retrieval_port,
                               nim_hallucination_port,
                               nim_answer_port,
                               nim_generator_id,
                               nim_router_id,
                               nim_retrieval_id,
                               nim_hallucination_id,
                               nim_answer_id,
                               chatbot], [msg, chatbot, actions]
        )

        sample_query_3.click(
            _my_build_stream, [sample_query_3, 
                               prompt_generator,
                               prompt_router,
                               prompt_retrieval,
                               prompt_hallucination,
                               prompt_answer,
                               nim_generator_ip,
                               nim_router_ip,
                               nim_retrieval_ip,
                               nim_hallucination_ip,
                               nim_answer_ip,
                               nim_generator_port,
                               nim_router_port,
                               nim_retrieval_port,
                               nim_hallucination_port,
                               nim_answer_port,
                               nim_generator_id,
                               nim_router_id,
                               nim_retrieval_id,
                               nim_hallucination_id,
                               nim_answer_id,
                               chatbot], [msg, chatbot, actions]
        )

        sample_query_4.click(
            _my_build_stream, [sample_query_4, 
                               prompt_generator,
                               prompt_router,
                               prompt_retrieval,
                               prompt_hallucination,
                               prompt_answer,
                               nim_generator_ip,
                               nim_router_ip,
                               nim_retrieval_ip,
                               nim_hallucination_ip,
                               nim_answer_ip,
                               nim_generator_port,
                               nim_router_port,
                               nim_retrieval_port,
                               nim_hallucination_port,
                               nim_answer_port,
                               nim_generator_id,
                               nim_router_id,
                               nim_retrieval_id,
                               nim_hallucination_id,
                               nim_answer_id,
                               chatbot], [msg, chatbot, actions]
        )
        
        msg.submit(
            _my_build_stream, [msg, 
                               prompt_generator,
                               prompt_router,
                               prompt_retrieval,
                               prompt_hallucination,
                               prompt_answer,
                               nim_generator_ip,
                               nim_router_ip,
                               nim_retrieval_ip,
                               nim_hallucination_ip,
                               nim_answer_ip,
                               nim_generator_port,
                               nim_router_port,
                               nim_retrieval_port,
                               nim_hallucination_port,
                               nim_answer_port,
                               nim_generator_id,
                               nim_router_id,
                               nim_retrieval_id,
                               nim_hallucination_id,
                               nim_answer_id,
                               chatbot], [msg, chatbot, actions]
        )

    page.queue()
    return page

""" This helper function verifies that a user query is nonempty. """

def valid_input(query: str):
    return False if query.isspace() or query is None or query == "" or query == '' else True


""" This helper function provides error outputs for the query. """
def _get_query_error_message(e: Exception) -> str:
    if isinstance(e, GraphRecursionError):
        err = QUERY_ERROR_MESSAGES["GraphRecursionError"]
    elif isinstance(e, HTTPError):
        if e.response is not None and e.response.status_code == 401:
            err = QUERY_ERROR_MESSAGES["AuthenticationError"]
        else:
            err = QUERY_ERROR_MESSAGES["HTTPError"]
    elif isinstance(e, TavilyAPIError):
        err = QUERY_ERROR_MESSAGES["TavilyAPIError"]
    else:
        err = QUERY_ERROR_MESSAGES["Unknown"]

    return f"{err['title']}\n\n{err['body']}"




""" This helper function executes and generates a response to the user query. """
def _stream_predict(
    client: chat_client.ChatClient,
    app, 
    question: str,
    prompt_generator: str,
    prompt_router: str,
    prompt_retrieval: str,
    prompt_hallucination: str,
    prompt_answer: str,
    nim_generator_ip: str,
    nim_router_ip: str,
    nim_retrieval_ip: str,
    nim_hallucination_ip: str,
    nim_answer_ip: str,
    nim_generator_port: str,
    nim_router_port: str,
    nim_retrieval_port: str,
    nim_hallucination_port: str,
    nim_answer_port: str,
    nim_generator_id: str,
    nim_router_id: str,
    nim_retrieval_id: str,
    nim_hallucination_id: str,
    nim_answer_id: str,
    chat_history: List[Tuple[str, str]],
) -> Any:

    inputs = {"question": question, 
              "prompt_generator": prompt_generator, 
              "prompt_router": prompt_router, 
              "prompt_retrieval": prompt_retrieval, 
              "prompt_hallucination": prompt_hallucination, 
              "prompt_answer": prompt_answer, 
              "nim_generator_ip": nim_generator_ip,
              "nim_router_ip": nim_router_ip,
              "nim_retrieval_ip": nim_retrieval_ip,
              "nim_hallucination_ip": nim_hallucination_ip,
              "nim_answer_ip": nim_answer_ip,
              "nim_generator_port": nim_generator_port,
              "nim_router_port": nim_router_port,
              "nim_retrieval_port": nim_retrieval_port,
              "nim_hallucination_port": nim_hallucination_port,
              "nim_answer_port": nim_answer_port,
              "nim_generator_id": nim_generator_id,
              "nim_router_id": nim_router_id,
              "nim_retrieval_id": nim_retrieval_id,
              "nim_hallucination_id": nim_hallucination_id,
              "nim_answer_id": nim_answer_id}
    
    if not valid_input(question):
        yield "", chat_history + [[str(question), "*** ERR: Unable to process query. Query cannot be empty. ***"]], gr.update(show_label=False)
    else: 
        try:
            actions = {}
            config = RunnableConfig(recursion_limit=RECURSION_LIMIT)
            for output in app.stream(inputs, config=config):
                actions.update(output)
                yield "", chat_history + [[question, "Working on getting you the best answer..."]], gr.update(value=actions)
                for key, value in output.items():
                    final_value = value
            yield "", chat_history + [[question, final_value["generation"]]], gr.update(show_label=False)

        except Exception as e:
            traceback.print_exc()

            message = _get_query_error_message(e)

            yield "", chat_history + [[question, message]], gr.update(show_label=False)