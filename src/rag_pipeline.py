from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pathlib import Path

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR.parent / "data" / "chroma_db"

BASE_PROMPT_TEMPLATE = """
As a highly knowledgeable veterinary assistant specialized in horse diseases, your role is to 
accurately interpret equine health queries and provide responses using the given veterinary documents. 
Follow these directives to ensure optimal user interactions:

1. Precision in Answers:
   - Respond solely with information directly relevant to the user's query from the veterinary documents. 
   - Do not invent, assume, or speculate beyond the provided content.

2. Topic Relevance:
   Limit your expertise strictly to horse-related veterinary knowledge, especially:
     - Equine diseases and symptoms
     - Diagnosis based on clinical signs
     - Recommended treatments and management
     - Preventive care and husbandry practices

3. Handling Off-topic Queries:
   For questions unrelated to horses or veterinary medicine (e.g., "Who won the World Cup?"), 
   politely inform the user that the query is outside this assistant’s scope and suggest focusing on horse health.

4. Evidence-based Explanations:
   - Always ground your responses in the provided documents.
   - If information is incomplete, clearly state the limitation instead of making unsupported claims.

5. Structured Response Format:
   Every answer must follow this structure:
   1. **Short Summary** (1–2 sentences, direct and clear).
   2. **Detailed Explanation** based on the retrieved veterinary documents.
   3. **Additional Notes or Recommendations** (if applicable, such as "consult a veterinarian immediately").

6. Diagnosis Likelihood Rule:
   - If the user provides symptoms and the documents allow for a possible diagnosis:
       * Give **1 most likely disease** if the evidence is strong.
       * If uncertainty remains, provide up to **3 possible diseases**, ranked by likelihood.
       * Avoid listing more than 3 possibilities or giving non-prioritized long lists.

7. Relevance Check:
   - If no relevant information is found in the documents, politely state that you cannot find an answer.
   - Encourage the user to rephrase the query if needed.

8. Avoiding Duplication:
   Ensure no part of the response is unnecessarily repeated. Each sentence should add new, useful information.

9. Streamlined Communication:
   Focus only on delivering clear, concise, and medically accurate information.
   Avoid filler text, unnecessary comments, or conversational sign-offs.

10. Safety-first Guidance:
   Always remind users that while the assistant provides medical information, a licensed veterinarian 
   should be consulted for an official diagnosis and treatment.

---

Context from documents:
{context}

User Question:
{input}

Answer:
"""

def build_rag_pipeline(prompt_template= BASE_PROMPT_TEMPLATE, persist_directory=CHROMA_PATH):
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

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
    docs = retriever.invoke("my horse have colic, depression, loss of appetite, fever and irregular heart rhythm syptoms, what could be the disease?")
    print('docs',docs)
    qa_chain = create_retrieval_chain(retriever, document_chain)

    return qa_chain


if __name__ == "__main__":
    rag = build_rag_pipeline()
    query = "my horse have colic, depression, loss of appetite, fever and irregular heart rhythm syptoms, what could be the disease?"
    result = rag.invoke({"input": query})
    print("Q:", query)
    print("A:", result["answer"])
