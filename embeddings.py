import json
import faiss
import torch
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

def create_embeddings():
    with open("metadata.json", "r", encoding="utf-8") as f:
        segment_data = json.load(f)
    
    model_name = "/home/cong/workspace/chatbot-retrieval-based/kaggle/working/mlm_model/checkpoint-6996"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    batch_size = 32
    index = faiss.IndexFlatL2(768)  
    segment_ids = []

    for i in range(0, len(segment_data), batch_size):
        batch = segment_data[i:i + batch_size]
        contents = [seg["content"] for seg in batch]
        encoded = tokenizer(contents, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            embeddings = mean_pooling(model(**encoded), encoded["attention_mask"]).numpy()
        index.add(embeddings)
        segment_ids.extend([seg["segment_id"] for seg in batch])
        
    faiss.write_index(index, "faiss_index.bin")
    with open("segment_ids.json", "w", encoding="utf-8") as f:
        json.dump(segment_ids, f, ensure_ascii=False)

create_embeddings()