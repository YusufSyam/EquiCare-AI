from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pathlib import Path

from src.utils.constants import BASE_PROMPT_TEMPLATE


BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR.parent / "data" / "chroma_db"

def build_rag_pipeline(prompt_template= BASE_PROMPT_TEMPLATE, persist_directory=CHROMA_PATH):
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

   #  tokenizer = AutoTokenizer.from_pretrained(
   #      "distilgpt2",
   #      model_max_length=512,
   #      padding_side="left",
   #      truncation=True
   #  )   

   #  llm_pipeline = pipeline(
   #      "text-generation",
   #      model="distilgpt2",  
   #      tokenizer=tokenizer,
   #      device_map="auto" ,
   #      max_length=512, 
   #      max_new_tokens=200,
   #  )

    model_name = "tiiuae/Falcon3-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        load_in_8bit=True,
    )

    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=1024,
        temperature=0.3,
        do_sample=True
    )

    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "input"]
    )

    document_chain = create_stuff_documents_chain(llm, PROMPT)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    qa_chain = create_retrieval_chain(retriever, document_chain)

    return qa_chain


if __name__ == "__main__":
    rag = build_rag_pipeline()
    query = "my horse have colic, depression, loss of appetite, fever and irregular heart rhythm syptoms, what could be the disease?"
    result = rag.invoke({"input": query})
    print("Q:", query)
    print("A:", result["answer"])
