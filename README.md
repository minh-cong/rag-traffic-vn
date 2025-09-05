# Hệ thống RAG cho Thông tư pháp luật Việt Nam

Hệ thống RAG (Retrieval-Augmented Generation) cho các thông tư pháp luật Việt Nam, sử dụng kỹ thuật tìm kiếm ngữ nghĩa và trích xuất nâng cao.

## Demo Giao Diện

Dưới đây là UI khi người dùng hỏi một câu trong ứng dụng:

![Giao diện ứng dụng trả lời câu hỏi luật giao thông](images/ui-demo.jpg)

## Cấu trúc dự án

```
.
├── create_metadata.py           # Script trích xuất metadata từ các file PDF/DOCX
├── create_embeddings_fixed.py   # Script tạo vector embeddings không phụ thuộc vào sentence-transformers
├── retrieval.py                 # Script truy vấn thông tin dựa trên FAISS và metadata
├── multihop_rag.py              # Hệ thống MultiHop RAG
├── server.py                    # API server
├── update_huggingface_hub.py    # Script tự động kiểm tra và khắc phục lỗi huggingface_hub
├── run_processing.py            # Script chạy toàn bộ quy trình xử lý dữ liệu
├── requirements.txt             # Danh sách thư viện cần thiết
├── metadata.json                # Metadata được trích xuất từ các file thông tư
├── faiss_index.bin              # FAISS index lưu trữ vector embeddings
├── segment_ids.json             # Danh sách segment_id tương ứng với embeddings
└── Thongtu/                     # Thư mục chứa các file PDF/DOCX thông tư
```

## Cài đặt

### 1. Cài đặt các thư viện Python

```bash
pip install -r requirements.txt
```

### 2. Cài đặt Poppler (để xử lý PDF)

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

**Windows:**
Tải Poppler từ: https://github.com/oschwartz10612/poppler-windows
Sau đó thêm thư mục bin vào PATH.

### 3. Cài đặt Tesseract OCR (để trích xuất text từ ảnh)

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-vie
```

**Windows:**
Tải Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
Cài đặt gói tiếng Việt và thêm đường dẫn vào PATH.

## Khắc phục lỗi huggingface_hub

Nếu bạn gặp lỗi liên quan đến `huggingface_hub`, hãy chạy:

```bash
python update_huggingface_hub.py
```

Script này sẽ tự động kiểm tra và khắc phục lỗi liên quan đến `huggingface_hub.utils.hf_hub_url`.

Chi tiết xem trong [HUONG_DAN.md](./HUONG_DAN.md).

## Sử dụng

### 1. Quy trình xử lý dữ liệu tự động

Chạy script tự động hóa:

```bash
python run_processing.py
```

### 2. Xử lý dữ liệu thủ công

#### a. Trích xuất metadata

```bash
python create_metadata.py
```

#### b. Tạo vector embeddings

```bash
python create_embeddings_fixed.py
```

#### c. Kiểm tra hệ thống truy vấn

```bash
python retrieval.py
```

### 3. Chạy API server

```bash
python server.py
```

Server sẽ khởi động tại http://localhost:8000

### 4. Sử dụng MultiHop RAG trong code Python

```python
from multihop_rag import MultiHopRAG

# Khởi tạo RAG
rag = MultiHopRAG(use_llm=False)  # use_llm=True để sử dụng LLM

# Truy vấn
result = rag.answer_query("Quy định về tốc độ tối đa của xe máy là gì?")
print(result["answer"])

# Lấy tài liệu tham khảo
references = result["references"]
```

## Chú ý

- Hệ thống sử dụng mô hình `distiluse-base-multilingual-cased-v2` cho việc tạo embeddings, được triển khai trực tiếp thông qua thư viện `transformers`.
- OCR được sử dụng để xử lý các file PDF quét (scanned), do đó chất lượng có thể khác nhau tùy thuộc vào chất lượng ảnh.
- Đối với các file PDF mới, khuyến nghị sử dụng định dạng PDF có thể tìm kiếm (searchable PDF) để cải thiện hiệu suất trích xuất. 
