import json
import os
from serpapi import GoogleSearch
import numpy as np
import faiss
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer, util
<<<<<<< HEAD
from pydantic import BaseModel, Field
from datetime import datetime


llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key="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",
    model_name="google/gemma-3-4b-it:free",
)

llm_router = llm
llm_rewriting = llm
llm_response = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key="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",
    model_name="meta-llama/llama-3.3-70b-instruct:free",
)


model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

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


class WebSearcher:
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY") 
        self.llm_response = llm_response 

    def search(self, query, num_results=5):
        if self.google_api_key:
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
        
        else:
            raise ValueError("Thiếu GOOGLE_API_KEY. Hãy thiết lập API key trước khi gọi search.")
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
        prompt = ChatPromptTemplate.from_template(multi_prompt)
        chain = (prompt | llm_rewriting | StrOutputParser() | (lambda x: [q.strip('* ') for q in x.strip().split('*') if q.strip()]))
        return chain.invoke({"query": query})

    def retrieve(self, queries, k=5, similarity_threshold=0.5):
        results = {}
        scores_dict = {}
        for query in queries:
            query_embedding = self.model.encode(query, convert_to_numpy=True).astype(np.float32)
            query_embedding = np.expand_dims(query_embedding, axis=0)
            D, I = self.index.search(query_embedding, k)
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

    def get_results(self, query, similarity_threshold=0.5):
        processed_queries = self.process_query(query)
        results, scores = self.retrieve(processed_queries, k=10, similarity_threshold=similarity_threshold)

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


def route_query(query, chat_history=""):
    query_lower = query.lower()
    search_terms = ["tiền", "2025", "phạt", "tiền phạt", "mức phạt"]

    if any(term in query_lower for term in search_terms):
        return "search"
    
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


class EnhancedChatbot:
    def __init__(self, session_id=None):
        self.memory_manager = MemoryManager(session_id)
        self.web_searcher = WebSearcher()  
        self.vector_retriever = VectorRetriever()

    def process_message(self, user_message):
        self.memory_manager.add_user_message(user_message)
        chat_history = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
            for msg in self.memory_manager.get_chat_history()[-6:]
        ])
        destination = route_query(user_message, chat_history)

        print(f"Routing decision: {destination}")

        if destination == "retrieval":
            try:
                query_grouped_segments, scores = self.vector_retriever.get_results(user_message, similarity_threshold=0.50)

                if query_grouped_segments:
                    responses = []
                    for sub_query, segments in query_grouped_segments.items():
                        if segments:
                            context = "\n".join([f"Tài liệu {r['document_id']}: {r['content']}" for r in segments])
                            sub_response_prompt = f"Dựa trên:\n{context}\nTrả lời cho phần: '{sub_query}' ngắn gọn, nêu nguồn nếu có."
                            sub_response = llm_response.invoke(sub_response_prompt).content
                            responses.append(f"Về {sub_query}: {sub_response}")
                    if responses:
                        response = "\n\n".join(responses)
                    else:
                        search_results = self.web_searcher.search(user_message)
                        response = self.web_searcher.format_search_results(user_message, search_results)
                else:
                    search_results = self.web_searcher.search(user_message)
                    response = self.web_searcher.format_search_results(user_message, search_results)

            except Exception as e:
                print(f"Error in retrieval: {str(e)}")
                search_results = self.web_searcher.search(user_message)
                response = self.web_searcher.format_search_results(user_message, search_results) + "\n*Lưu ý: Thông tin từ web, kiểm tra văn bản pháp luật để chắc chắn.*"

        elif destination == "search":
            search_results = self.web_searcher.search(user_message)
            response = (
                self.web_searcher.format_search_results(user_message, search_results)
                if search_results else "Không tìm thấy thông tin trên web."
            )

        else:
            messages = self.memory_manager.get_chat_history()
            formatted_history = "\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in messages[-6:]
            ])
            conversation_prompt = (
                f"""
                Bạn là một trợ lý AI hữu ích và thân thiện có tên ChainedTogether. Dựa trên lịch sử cuộc trò chuyện sau:
                {formatted_history}
                Trả lời tin nhắn mới nhất của người dùng: {user_message}
                """
            )
            response = llm_response.invoke([
                SystemMessage(content="Bạn là một trợ lý AI hữu ích và thân thiện."),
                HumanMessage(content=conversation_prompt)
            ]).content

        self.memory_manager.add_ai_message(response)
        return {"response": response}

=======
from openai import OpenAI
import numpy as np
from langchain_core.pydantic_v1 import BaseModel, Field

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="",
)

# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0]
#     mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

class GradeDocuments(BaseModel):
    related: str = Field(
        description= "Documents are relevant to the question just answer, 'yes' or 'no'"
    )

class VectorRetriever:
    def __init__(self):
        print("Đang khởi tạo VectorRetriever...")
        model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = SentenceTransformer(model_name)
        with open("metadata.json", "r", encoding="utf-8") as f:
            self.segment_data = json.load(f)
        with open("segment_ids.json", "r", encoding="utf-8") as f:
            self.segment_ids = json.load(f)
        self.index = faiss.read_index("faiss_index.bin")
        print("Đã tải xong dữ liệu!")
    def process_query(self,query):
        prompt = f"""Viết lại query sau đây của người dùng thành một yêu cầu rõ ràng, cụ thể và trang trọng bằng tiếng Việt, phù hợp để truy xuất thông tin từ một cơ sở dữ liệu vector.
    Nếu như query có liên quan đến nhiều thông tin hãy tách ra làm nhiều query mới để có thể truy xuất nhiều lần. Với mỗi query hãy cho bắt đầu và kết thúc nằm ở trong '*'.
    Chỉ trả về query mới, không cần giải thích gì thêm
    Lưu ý rằng query mới sẽ được gửi đến cơ sở dữ liệu vector, nơi thực hiện tìm kiếm tương đồng để truy xuất tài liệu. Ví dụ:  
    Query: Chúng tôi có một bài luận phải nộp vào ngày mai. Chúng tôi phải viết về một số loài động vật. Tôi yêu chim cánh cụt. Tôi có thể viết về chúng. Nhưng tôi cũng có thể viết về cá heo. Chúng có phải là động vật không? Có lẽ vậy. Hãy viết về cá heo. Ví dụ, chúng sống ở đâu?  
    Answer: * Cá heo sống ở đâu *
    Ví dụ:  
    Query: So sánh doanh thu của FPT và Viettel
    Answer : * Doanh thu của FPT *,* Doanh thu Viettel*
    Bây giờ, hãy viết lại query sau: Query: {query} Answer:"""
        completion = client.chat.completions.create(
            model="google/gemma-3-27b-it:free",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        results = completion.choices[0].message.content
        queries = [line.strip('* ').strip() for line in results.splitlines() if line.strip()]
        return queries
    def encode(self, queries):
        queries_embedding = []
        for query in queries:
            query_embedding = self.model.encode(query)
            query_embedding = query_embedding.astype(np.float32)
            queries_embedding.append(query_embedding)
        return queries_embedding

    def retrieve(self, queries, k=10):
        print(f"Đang tìm kiếm với truy vấn: {queries}")
        queries_embedding = self.encode(queries)
        results = {}
        for idx, query in enumerate(queries):
            embedding = np.expand_dims(queries_embedding[idx], axis=0)
            D, I = self.index.search(embedding, k)
            print(f"Truy vấn '{query}' tìm thấy {len(I[0])} kết quả thô")
            valid_indices = [i for i in I[0] if i < len(self.segment_data)]
            results[query] = [self.segment_data[i] for i in valid_indices]
        return results

    def filter_results(self, query, segments, threshold=0):
        query_embedding = self.model.encode(query)
        filtered_segments = []
        for seg in segments:
            seg_embedding = self.model.encode(seg["content"])
            similarity = util.pytorch_cos_sim(query_embedding, seg_embedding).item()
            if similarity >= threshold:
                filtered_segments.append(seg)
                print(seg)
        return filtered_segments

def generate_response(query, context,temperature = 0.1):
    """Sử dụng OpenRouter để tổng hợp câu trả lời từ context"""
    prompt = (
        f"Dựa trên thông tin sau đây:\n\n{context}\n\n"
        f"""Hãy trả lời câu hỏi: '{query}' một cách rõ ràng nêu xem điều bạn vừa nói trích xuất ở luật nào,
        Nếu như người dùng có phạm tội thì hãy phân tích những tội đó ở đâu. Trả lời một cách ngắn gọn, trang trọng
        Ví dụ:  
        Query: Tốc độ tối đa của xe máy là bao nhiêu 
        Answer: Dựa theo điều 8, etc, tốc độ tối đa của xe máy là
        Dựa vào mẫu trên cùng với content hãy trả lời query của người dùng
        """
    )
    completion = client.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct:free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature
    )
    return completion.choices[0].message.content
>>>>>>> eb9c4826f84c6efb2b530567b0fc5185b7e28f79


if __name__ == "__main__":
<<<<<<< HEAD
    chatbot = EnhancedChatbot()
    print("Enhanced Chatbot ready. Type 'exit' to quit.")
    print("="*50)
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
        result = chatbot.process_message(user_input)
        print(f"AI: {result['response']}")
        print("="*50)
=======
    try:
        retriever = VectorRetriever()
        query = input("Nhập câu hỏi của bạn: ")
        processed_queries = retriever.process_query(query)
        results = retriever.retrieve(processed_queries, k=5)
        
        all_segments = []
        for seg_list in results.values():
            all_segments.extend(seg_list)
        
        filtered_results = []
        for q in processed_queries:
            filtered = retriever.filter_results(q, all_segments, threshold=0.55)
            filtered_results.extend(filtered)
        
        if filtered_results:
            context = "\n".join([str('thông tin về content' + res["document_id"] + res["content"]) for res in filtered_results])
            # print("\n=== KẾT QUẢ TÌM KIẾM ===")
            # for i, res in enumerate(filtered_results, 1):
            #     print(f"{format_segment(res)}")
            
            response = generate_response(query, context)
            print("\n=== CÂU TRẢ LỜI TỔNG HỢP ===")
            print(response)
        else:
            print("Không tìm thấy kết quả nào phù hợp.")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
>>>>>>> eb9c4826f84c6efb2b530567b0fc5185b7e28f79
