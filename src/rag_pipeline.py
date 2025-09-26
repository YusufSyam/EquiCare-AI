from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pathlib import Path

from src.utils.string_constants import BASE_PROMPT_TEMPLATE
from src.utils.model_config import MODEL_CONFIG

embedding_model_name = MODEL_CONFIG["embedding_model_name"]
text_generation_model_name = MODEL_CONFIG["text_generation_model_name"]
k = MODEL_CONFIG["k"]
max_new_tokens = MODEL_CONFIG["max_new_tokens"]
device_map = MODEL_CONFIG["device_map"]


BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR.parent / "data" / "chroma_db"

def build_rag_pipeline(prompt_template= BASE_PROMPT_TEMPLATE, persist_directory=CHROMA_PATH):
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name
    )

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    tokenizer = AutoTokenizer.from_pretrained(text_generation_model_name)

    model = AutoModelForCausalLM.from_pretrained(
        text_generation_model_name,
        device_map="auto",
        torch_dtype="auto",
        load_in_8bit=True,
    )

    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=device_map,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        do_sample=True
    )

    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "input"]
    )

    document_chain = create_stuff_documents_chain(llm, PROMPT)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    
    qa_chain = create_retrieval_chain(retriever, document_chain)

    return qa_chain


if __name__ == "__main__":
    rag = build_rag_pipeline()
    query = "my horse have colic, depression, loss of appetite, fever and irregular heart rhythm syptoms, what could be the disease?"
    result = rag.invoke({"input": query})
    print("Q:", query)
    print("A:", result["answer"])
