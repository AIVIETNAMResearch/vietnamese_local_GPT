import os
import logging
# import click
import torch
import utils
import streamlit as st

from langdetect import detect

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager
from nlp_preprocessing import Translation

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from prompt_template_utils import get_prompt_template

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    GenerationConfig,
    pipeline,
)

from load_models import (
    load_quantized_model_awq,
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS
)


def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            print("Load quantized model gguf")
            llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            return llm
        elif ".ggml" in model_basename.lower():
            print("Load quantized model ggml")
            model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
        elif ".awq" in model_basename.lower():
            print("Load quantized model awq")
            model, tokenizer = load_quantized_model_awq(model_id, LOGGING)
        else:
            print("Load quantized model qptq")
            model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
    else:
        print("load_full_model")
        model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_NEW_TOKENS,
        temperature=0.2,
        # top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm


def retrieval_qa_pipline(device_type, use_history, promptTemplate_type="llama"):
    """
    Initializes and returns a retrieval-based Question Answering (QA) pipeline.

    This function sets up a QA system that retrieves relevant information using embeddings
    from the HuggingFace library. It then answers questions based on the retrieved information.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'cuda', etc.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Returns:
    - RetrievalQA: An initialized retrieval-based QA system.

    Notes:
    - The function uses embeddings from the HuggingFace library, either instruction-based or regular.
    - The Chroma class is used to load a vector store containing pre-computed embeddings.
    - The retriever fetches relevant documents or data based on a query.
    - The prompt and memory, obtained from the `get_prompt_template` function, might be used in the QA system.
    - The model is loaded onto the specified device using its ID and basename.
    - The QA system retrieves relevant documents using the retriever and then answers questions based on those documents.
    """

    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})
    # uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # load the vectorstore
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS
    )
    retriever = db.as_retriever()

    # get the prompt template and memory if set by the user.
    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)

    # load the llm pipeline
    llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)

    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )

    return qa


# chose device typ to run on as well as to show source documents.
# @click.command()
# @click.option(
#     "--device_type",
#     default="cuda" if torch.cuda.is_available() else "cpu",
#     type=click.Choice(
#         [
#             "cpu",
#             "cuda",
#             "ipu",
#             "xpu",
#             "mkldnn",
#             "opengl",
#             "opencl",
#             "ideep",
#             "hip",
#             "ve",
#             "fpga",
#             "ort",
#             "xla",
#             "lazy",
#             "vulkan",
#             "mps",
#             "meta",
#             "hpu",
#             "mtia",
#         ],
#     ),
#     help="Device to run on. (Default is cuda)",
# )
# @click.option(
#     "--show_sources",
#     "-s",
#     is_flag=True,
#     help="Show sources along with answers (Default is False)",
# )
# @click.option(
#     "--use_history",
#     "-h",
#     is_flag=True,
#     help="Use history (Default is False)",
# )
# @click.option(
#     "--model_type",
#     default="llama",
#     type=click.Choice(
#         ["llama", "mistral", "non_llama"],
#     ),
#     help="model type, llama, mistral or non_llama",
# )
# @click.option(
#     "--save_qa",
#     is_flag=True,
#     help="whether to save Q&A pairs to a CSV file (Default is False)",
# )

# @click.option(
#     "--translate_output",
#     "-t",
#     is_flag=True,
#     help="translate answer to VN lang",
# )

def get_response(query, qa_pipeline):
    res = qa_pipeline(query)
    answer, docs = res["result"], res["source_documents"]

    # translate answer to VN
    # if translate_output:
    #     if detect(answer) != 'vi':
    #         ans_lang = detect(answer)
    #         translater = Translation(from_lang=ans_lang, to_lang='vi', mode='translate') 
    #         answer = translater(answer)

    return answer

if __name__ == "__main__":
    device_type = "cpu"
    use_history = False 
    model_type = "llama"
    translate_output = False 
    qa = retrieval_qa_pipline(device_type, use_history, promptTemplate_type=model_type)

    st.title("💬 Chatbot")
    st.caption("Local LLM")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        print("prompt: ", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = get_response(prompt, qa)
        msg = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)

