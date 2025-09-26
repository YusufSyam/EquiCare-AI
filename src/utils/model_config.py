MODEL_CONFIG = {
    "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "text_generation_model_name": "tiiuae/Falcon3-1B-Instruct", # meta-llama/Llama-2-7b-chat-hf | distilgpt2
    "k": 3,  # number of chunks retrieved from vectordb
    "max_new_tokens": 1024,
    "device_map": "auto",
}

# MODEL_CONFIG = {
#     "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
#     "text_generation_model_name": "distilgpt2",
#     "k": 2,  # number of chunks retrieved from vectordb
#     "max_new_tokens": 512,
#     "device_map": "auto",
# }