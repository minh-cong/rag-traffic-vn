from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import faiss
import json
import os
from langchain_openai import ChatOpenAI  
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sentence_transformers import SentenceTransformer, util
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory  
from langchain_core.messages import HumanMessage, SystemMessage
from datetime import datetime
from pydantic import Field
from serpapi import GoogleSearch 


client = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key="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",
    model_name="meta-llama/llama-3.3-70b-instruct:free",
)

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


class RouterOutput(BaseModel):
    destination: str = Field(description="Route to: 'search', 'retrieval', or 'conversation'")
    reason: str = Field(description="Reason for routing")


class MemoryManager:
    def __init__(self, session_id=None):
        self.session_id = session_id or str(datetime.now().timestamp())
        self.memory = ConversationBufferMemory(return_messages=True)
    
    def add_user_message(self, message: str):
        self.memory.chat_memory.add_user_message(message)
    
    def add_ai_message(self, message: str):
        self.memory.chat_memory.add_ai_message(message)
    
    def get_chat_history(self):
        return self.memory.load_memory_variables({}).get("history", [])
    
    def clear_history(self):
        self.memory = ConversationBufferMemory(return_messages=True)


class WebSearcher:
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY") 
        self.llm_response = ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key="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",
            model_name="meta-llama/llama-3.3-70b-instruct:free",
        )
    
    def search(self, query, num_results=5):
        if not self.google_api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY in your environment.")

        params = {
            "q": f"{query} 'Nghị định 168/2024/NĐ-CP' site:*.gov.vn site:thuvienphapluat.vn -inurl:(signup login)",
            "api_key": self.google_api_key,
            "num": num_results,
            "gl": "vn",  
            "hl": "vi"   
        }
        search = GoogleSearch(params)
        results = search.get_dict().get("organic_results", [])
        parsed_results = [
            {"title": r.get("title", ""), "snippet": r.get("snippet", ""), "link": r.get("link", "")}
            for r in results[:num_results]
        ]

        if not parsed_results:
            print("No prioritized results, trying broader search.")
            params["q"] = query 
            search = GoogleSearch(params)
            results = search.get_dict().get("organic_results", [])
            parsed_results = [
                {"title": r.get("title", ""), "snippet": r.get("snippet", ""), "link": r.get("link", "")}
                for r in results[:num_results]
            ]
        
        return parsed_results
    
    def format_search_results(self, query, results):
        context = "\n".join([f"{r['title']}: {r['snippet']} (Nguồn: {r['link']})" for r in results])
        summary_prompt = ChatPromptTemplate.from_template("""
        Dựa trên thông tin sau:
        {context}
        
        Tóm tắt câu trả lời cho câu hỏi: "{query}" một cách ngắn gọn, rõ ràng, và chỉ giữ lại thông tin liên quan.
        
        - Đảm bảo phân biệt rõ mức phạt cho ô tô và xe máy (ô tô thường phạt nặng hơn).
        - Ưu tiên thông tin từ Nghị định 168/2024/NĐ-CP nếu có (hiệu lực từ 1/1/2025).
        - Nếu liên quan xe máy trên cao tốc, ghi rõ xe máy bị cấm (Nghị định 168/2024/NĐ-CP, Điều 7).
        - Nêu nguồn nếu có, ưu tiên thuvienphapluat.vn hoặc .gov.vn.
        - Nếu dùng web search, thêm: "*Lưu ý: Thông tin từ web, kiểm tra văn bản pháp luật để chắc chắn.*"
        - Nếu không có thông tin cụ thể, nói: "Không tìm thấy thông tin chi tiết về mức phạt."
        """)
        chain = summary_prompt | self.llm_response | StrOutputParser()
        response = chain.invoke({"context": context, "query": query})
        if not results:
            response += "\nKhông tìm thấy thông tin trên web, có thể cần kiểm tra trực tiếp Nghị định 168/2024/NĐ-CP."
        return response


class VectorRetriever:
    def __init__(self, index_file="faiss_index.bin", metadata_file="metadata.json"):
        print("Starting VectorRetriever...")
        self.model_name = "sentence-transformers/distiluse-base-multilingual-cased-v2"
        self.model = SentenceTransformer(self.model_name)

        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file {metadata_file} not found.")
        with open(metadata_file, "r", encoding="utf-8") as f:
            self.segment_data = json.load(f)
        self.documents = [Document(page_content=seg["content"], metadata=seg) for seg in self.segment_data]
        
        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)
            print("Loaded existing FAISS index.")
        else:
            print("Building new FAISS index...")
            document_embeddings = np.array([self.model.encode(doc.page_content, convert_to_numpy=True) for doc in self.documents], dtype=np.float32)
            embedding_dim = document_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(embedding_dim)
            self.index.add(document_embeddings)
            faiss.write_index(self.index, index_file)
            print("New index created and saved.")
        print("Data loaded!")

    def process_query(self, query):
        multi_prompt = """
        Analyze this query: "{query}"
        If it contains multiple questions or aspects (like asking about both cars AND motorcycles), 
        rewrite it as multiple separate Vietnamese queries for vector database retrieval.
        Format each query surrounded by '*'. For example: *Query about cars* *Query about motorcycles*
        If it's a single question, just rewrite it as one clear query surrounded by '*'.
        Answer:
        """
        llm_rewriting = ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key="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",
            model_name="meta-llama/llama-3.3-70b-instruct:free",
        )
        prompt = ChatPromptTemplate.from_template(multi_prompt)
        chain = (prompt | llm_rewriting | StrOutputParser() | (lambda x: [q.strip('* ') for q in x.strip().split('*') if q.strip()]))
        return chain.invoke({"query": query})

    def retrieve(self, queries, k=5, similarity_threshold=0.7):
        results = {}
        scores_dict = {}
        for query in queries:
            query_embedding = self.model.encode(query, convert_to_numpy=True).astype(np.float32)
            query_embedding = np.expand_dims(query_embedding, axis=0)
            D , I = self.index.search(query_embedding, k)
            segments = [self.segment_data[i] for i in I[0] if i < len(self.segment_data)]
            
            query_vec = self.model.encode(query)
            filtered_segments = []
            scores = []
            for seg in segments:
                seg_vec = self.model.encode(seg["content"])
                similarity = util.pytorch_cos_sim(query_vec, seg_vec).item()
                if similarity >= similarity_threshold:
                    filtered_segments.append(seg)
                    scores.append(similarity)
            
            results[query] = filtered_segments
            scores_dict[query] = scores
        
        return results, scores_dict


    def get_results(self, query, similarity_threshold=None):
        processed_queries = self.process_query(query)
        results, scores = self.retrieve(processed_queries, k=10, similarity_threshold=similarity_threshold or 0.7)
        
        all_segments = []
        all_scores = []
        query_sources = []
        
        for q_idx, (q, seg_list) in enumerate(results.items()):
            q_scores = scores[q]
            for idx, seg in enumerate(seg_list):
                if idx < len(q_scores):
                    seg["similarity_score"] = q_scores[idx]
                    seg["source_query"] = q
                    all_segments.append(seg)
                    all_scores.append(q_scores[idx])
                    query_sources.append(q)
        
        unique_segments = {}
        unique_scores = {}
        unique_queries = {}
        
        for idx, seg in enumerate(all_segments):
            unique_id = seg["document_id"] + "-" + seg["segment_id"]
            if unique_id not in unique_segments or all_scores[idx] > unique_scores[unique_id]:
                unique_segments[unique_id] = seg
                unique_scores[unique_id] = all_scores[idx]
                unique_queries[unique_id] = query_sources[idx]
        
        query_grouped_segments = {}
        for unique_id, seg in unique_segments.items():
            source_query = unique_queries[unique_id]
            if source_query not in query_grouped_segments:
                query_grouped_segments[source_query] = []
            query_grouped_segments[source_query].append(seg)
        
        return query_grouped_segments, unique_scores


def route_query(query, chat_history="", llm_router=None):
    query_lower = query.lower()
    search_terms = ["tiền", "2025", "phạt", "tiền phạt", "mức phạt"]
    retrieval_terms = ["luật", "nghị định", "điều khoản", "thông tư", "quy định", "tốc độ tối đa", "biển báo giao thông"]

    if any(term in query_lower for term in search_terms):
        return "search"

    if any(term in query_lower for term in retrieval_terms) and not any(term in query_lower for term in search_terms):
        return "retrieval"

    if len(query.split()) <= 5 and not any(term in query_lower for term in search_terms + retrieval_terms):
        return "conversation"

    router_prompt = ChatPromptTemplate.from_template("""
    Phân tích truy vấn người dùng và lịch sử trò chuyện:
    Chat history: {chat_history}
    Query: {query}
    Quyết định hệ thống nào nên xử lý:
    - "retrieval": nếu hỏi về luật hoặc tài liệu trong cơ sở dữ liệu (không nhắc đến 2025)
    - "search": nếu cần thông tin hiện tại từ web (như luật 2025, mức phạt, tốc độ mới)
    - "conversation": nếu là trò chuyện chung hoặc dựa trên lịch sử
    Trả về: destination (một trong "retrieval", "search", "conversation")
    """)
    chain = router_prompt | llm_router | StrOutputParser()
    response = chain.invoke({"query": query, "chat_history": chat_history})

    if "retrieval" in response.lower():
        return "retrieval"
    elif "search" in response.lower():
        return "search"
    else:
        return "conversation"


memory_manager = MemoryManager()
web_searcher = WebSearcher()
retriever = VectorRetriever()


@app.get("/", response_class=FileResponse)
async def read_index():
    return FileResponse("index.html")

@app.post("/search")
async def search(query_request: QueryRequest):
    query = query_request.query
    memory_manager.add_user_message(query)
    chat_history = "\n".join([
        f"{'User' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
        for msg in memory_manager.get_chat_history()[-6:]
    ])
    destination = route_query(query, chat_history)
    print(f"Routing decision: {destination}")

    llm_response = ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key="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",
        model_name="meta-llama/llama-3.3-70b-instruct:free",
    )

    try:
        if destination == "retrieval":
            query_grouped_segments, scores = retriever.get_results(query)
            if query_grouped_segments:
                responses = []
                for sub_query, segments in query_grouped_segments.items():
                    if segments:
                        context = "\n".join([f"Tài liệu {r['document_id']}: {r['content']}" for r in segments])
                        sub_response_prompt = f"Dựa trên:\n{context}\nTrả lời cho phần: '{sub_query}' ngắn gọn, nêu nguồn nếu có."
                        sub_response = llm_response.invoke(sub_response_prompt).content
                        responses.append(f"Về {sub_query}: {sub_response}")
                response = "\n\n".join(responses) if responses else "Không tìm thấy thông tin chi tiết trong tài liệu."
            else:
                search_results = web_searcher.search(query)
                response = web_searcher.format_search_results(query, search_results)
        
        elif destination == "search":
            search_results = web_searcher.search(query)
            response = web_searcher.format_search_results(query, search_results) if search_results else "Không tìm thấy thông tin trên web."

        else: 
            messages = memory_manager.get_chat_history()
            formatted_history = "\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in messages[-6:]
            ])
            conversation_prompt = f"Lịch sử:\n{formatted_history}\nTrả lời: {query}"
            response = llm_response.invoke([
                SystemMessage(content="Bạn là ChainedTogether, một Trợ Lý Pháp Luật hữu ích."),
                HumanMessage(content=conversation_prompt)
            ]).content

        memory_manager.add_ai_message(response)
        return {"results": [], "answer": response}

    except Exception as e:
        error_msg = f"Lỗi: {str(e)}. Vui lòng thử lại."
        memory_manager.add_ai_message(error_msg)
        return {"results": [], "answer": error_msg}

@app.post("/clear_history")
async def clear_history():
    memory_manager.clear_history()
    return {"message": "Lịch sử trò chuyện đã được xóa."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)