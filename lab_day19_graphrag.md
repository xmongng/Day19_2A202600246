# LAB DAY 19: XÂY DỰNG HỆ THỐNG GRAPHRAG VỚI TECH COMPANY CORPUS

---

## 1. MỤC TIÊU BÀI HỌC

- Hiểu rõ quy trình **trích xuất thực thể** (Entity Extraction) và **quan hệ** (Relation Extraction) từ văn bản thô.
- Làm quen với các thư viện quản lý đồ thị: **NetworkX**, **Neo4j** và framework mã nguồn mở **NodeRAG**.
- Xây dựng hoàn chỉnh một pipeline GraphRAG: từ **lập chỉ mục** (Indexing) đến **truy vấn đa bước** (Multi-hop Querying).
- Đánh giá sự khác biệt về độ chính xác giữa **Flat RAG** và **GraphRAG**.

---

## 2. PHẦN 1: NGHIÊN CỨU VÀ CHUẨN BỊ (RESEARCH)

Trước khi bắt đầu code, sinh viên cần tìm hiểu các khái niệm và công cụ sau:

### 2.1. Quy trình xử lý dữ liệu đồ thị

Sinh viên cần trả lời được các câu hỏi:

1. **Entity Extraction:** Làm sao để LLM phân biệt được đâu là thực thể (Node) và đâu là thuộc tính?
2. **Graph Construction:** Tại sao việc khử trùng lặp (Deduplication) lại quan trọng trong đồ thị?
3. **Query Answering:** Sự khác biệt giữa duyệt đồ thị theo chiều rộng (BFS) và tìm kiếm vector thông thường là gì?

### 2.2. Tìm hiểu công cụ

| Công cụ | Mô tả |
|---|---|
| **NetworkX** | Thư viện Python dùng để nghiên cứu các mạng lưới phức tạp. Phù hợp cho việc tạo prototype nhanh. |
| **Neo4j** | Cơ sở dữ liệu đồ thị chuẩn công nghiệp, sử dụng ngôn ngữ truy vấn Cypher. |
| **NodeRAG** | Một framework mã nguồn mở xây dựng trên nền NetworkX, giúp đơn giản hóa việc tích hợp GraphRAG vào ứng dụng Python. |

---

## 3. PHẦN 2: ENVIRONMENT SETUP

Mở terminal hoặc command prompt và cài đặt các thư viện cần thiết:

```bash
# Cài đặt các thư viện cơ bản cho xử lý ngôn ngữ và đồ thị
pip install networkx matplotlib neo4j openai pandas

# Cài đặt NodeRAG framework
pip install noderag

# Nếu sử dụng LangChain để hỗ trợ pipeline
pip install langchain langchain-openai
```

> **Lưu ý:** Đối với Neo4j, sinh viên nên sử dụng **Neo4j Desktop** hoặc chạy qua **Docker** để có giao diện trực quan hóa (Bloom/Browser).

```bash
# Chạy Neo4j bằng Docker
docker run \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

---

## 4. PHẦN 3: HƯỚNG DẪN THỰC HIỆN TỪNG BƯỚC

### Bước 1: Trích xuất thực thể và quan hệ (Indexing)

Sử dụng LLM để đọc bộ dữ liệu "Tech Company Corpus" và chuyển đổi thành các bộ ba (Triples).

**Ví dụ:**
- **Input:** `"OpenAI được thành lập bởi Sam Altman và Elon Musk vào năm 2015."`
- **Output (Triples):**
  - `(OpenAI, FOUNDED_BY, Sam Altman)`
  - `(OpenAI, FOUNDED_BY, Elon Musk)`
  - `(OpenAI, FOUNDED_IN, 2015)`

**Code mẫu — Entity Extraction bằng OpenAI:**

```python
from openai import OpenAI

client = OpenAI()

def extract_triples(text: str) -> list[dict]:
    prompt = f"""
Bạn là một hệ thống trích xuất tri thức. 
Hãy trích xuất tất cả các bộ ba (subject, relation, object) từ đoạn văn sau.
Trả về dưới dạng JSON list.

Văn bản: {text}

Ví dụ output:
[
  {{"subject": "OpenAI", "relation": "FOUNDED_BY", "object": "Sam Altman"}},
  {{"subject": "OpenAI", "relation": "FOUNDED_IN", "object": "2015"}}
]
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    import json
    return json.loads(response.choices[0].message.content)

# Test
text = "OpenAI được thành lập bởi Sam Altman và Elon Musk vào năm 2015."
triples = extract_triples(text)
print(triples)
```

---

### Bước 2: Xây dựng đồ thị (Construction)

Sinh viên chọn **một trong ba** công cụ sau:

#### Lựa chọn A — NetworkX (offline, phù hợp Notebook)

```python
import networkx as nx
import matplotlib.pyplot as plt

# Khởi tạo đồ thị có hướng
G = nx.DiGraph()

# Thêm các triple vào đồ thị
triples = [
    ("OpenAI", "Sam Altman", {"relation": "FOUNDED_BY"}),
    ("OpenAI", "Elon Musk",  {"relation": "FOUNDED_BY"}),
    ("OpenAI", "2015",       {"relation": "FOUNDED_IN"}),
    ("Google", "Larry Page",  {"relation": "FOUNDED_BY"}),
    ("Google", "Sergey Brin", {"relation": "FOUNDED_BY"}),
    ("Google", "1998",        {"relation": "FOUNDED_IN"}),
]

for subj, obj, attrs in triples:
    G.add_edge(subj, obj, **attrs)

# Khử trùng lặp — NetworkX tự động xử lý (nodes/edges là unique)
print(f"Số nodes: {G.number_of_nodes()}")
print(f"Số edges: {G.number_of_edges()}")

# Visualize
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue',
        node_size=2000, font_size=10, arrows=True)
edge_labels = nx.get_edge_attributes(G, 'relation')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Tech Company Knowledge Graph")
plt.savefig("knowledge_graph.png", dpi=150, bbox_inches='tight')
plt.show()
```

#### Lựa chọn B — Neo4j (trực quan hóa bằng Browser)

```python
from neo4j import GraphDatabase

URI      = "bolt://localhost:7687"
USER     = "neo4j"
PASSWORD = "password"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def create_triple(tx, subject, relation, obj):
    query = """
    MERGE (s:Entity {name: $subject})
    MERGE (o:Entity {name: $obj})
    MERGE (s)-[r:RELATION {type: $relation}]->(o)
    """
    tx.run(query, subject=subject, relation=relation, obj=obj)

triples_data = [
    ("OpenAI", "FOUNDED_BY", "Sam Altman"),
    ("OpenAI", "FOUNDED_BY", "Elon Musk"),
    ("OpenAI", "FOUNDED_IN", "2015"),
]

with driver.session() as session:
    for subj, rel, obj in triples_data:
        session.execute_write(create_triple, subj, rel, obj)

print("✅ Đã lưu đồ thị vào Neo4j!")
driver.close()
```

Sau đó mở **Neo4j Browser** tại `http://localhost:7474` và chạy Cypher:

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100
```

#### Lựa chọn C — NodeRAG (all-in-one, đã tối ưu logic tìm kiếm)

```python
# NodeRAG tích hợp sẵn GraphRAG pipeline
from noderag import NodeRAG

rag = NodeRAG()

# Indexing từ corpus text
corpus = [
    "OpenAI được thành lập bởi Sam Altman và Elon Musk vào năm 2015.",
    "Google được thành lập bởi Larry Page và Sergey Brin vào năm 1998.",
    "Microsoft được thành lập bởi Bill Gates và Paul Allen vào năm 1975.",
]

rag.index(corpus)
print("✅ Đã index xong!")
```

---

### Bước 3: Thực thi truy vấn (Querying)

Viết hàm xử lý truy vấn theo logic:

1. Nhận câu hỏi từ người dùng.
2. Trích xuất thực thể chính trong câu hỏi (ví dụ: `"Google"`).
3. Tìm node tương ứng trong đồ thị và duyệt (traverse) các node lân cận trong phạm vi **2-hop**.
4. Gộp các thông tin tìm được thành một đoạn văn (Textualization) và gửi cho LLM.

```python
def graphrag_query(G: nx.DiGraph, question: str, entity: str, hops: int = 2) -> str:
    """
    Multi-hop query trên đồ thị NetworkX.
    """
    if entity not in G.nodes:
        return f"Không tìm thấy entity '{entity}' trong đồ thị."

    # BFS để lấy tất cả node trong phạm vi `hops`
    subgraph_nodes = set()
    frontier = {entity}
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            neighbors = set(G.successors(node)) | set(G.predecessors(node))
            next_frontier.update(neighbors)
        subgraph_nodes.update(frontier)
        frontier = next_frontier - subgraph_nodes
    subgraph_nodes.update(frontier)

    # Textualization — chuyển subgraph thành văn bản
    facts = []
    subG = G.subgraph(subgraph_nodes)
    for u, v, data in subG.edges(data=True):
        relation = data.get("relation", "RELATED_TO")
        facts.append(f"{u} --[{relation}]--> {v}")

    context = "\n".join(facts)

    # Gửi context + question cho LLM
    prompt = f"""
Dựa trên các thông tin sau từ Knowledge Graph:
{context}

Hãy trả lời câu hỏi: {question}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Test
answer = graphrag_query(G, "Ai thành lập OpenAI?", entity="OpenAI")
print(answer)
```

---

### Bước 4: So sánh và Đánh giá (Evaluation)

Sinh viên chạy thử **5 câu hỏi phức tạp** trên cả hai hệ thống:

| STT | Câu hỏi | Flat RAG | GraphRAG | Ghi chú |
|---|---|---|---|---|
| 1 | Ai thành lập OpenAI? | | | |
| 2 | Công ty nào được thành lập trước: Google hay Microsoft? | | | |
| 3 | Sam Altman liên quan đến tổ chức nào? | | | |
| 4 | Các công ty thành lập sau năm 2000 là gì? | | | |
| 5 | Elon Musk có liên kết với công ty nào? | | | |

**Yêu cầu:** Ghi lại các trường hợp Flat RAG bị **ảo giác (hallucination)** nhưng GraphRAG trả lời đúng.

**Code đánh giá Flat RAG (baseline):**

```python
import chromadb
from chromadb.utils import embedding_functions

# Setup ChromaDB
chroma_client = chromadb.Client()
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-api-key",
    model_name="text-embedding-3-small"
)

collection = chroma_client.create_collection(
    name="tech_corpus",
    embedding_function=embedding_fn
)

# Index corpus
collection.add(
    documents=corpus,
    ids=[f"doc_{i}" for i in range(len(corpus))]
)

def flat_rag_query(question: str, n_results: int = 3) -> str:
    results = collection.query(query_texts=[question], n_results=n_results)
    context = "\n".join(results["documents"][0])
    prompt = f"Context:\n{context}\n\nCâu hỏi: {question}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

---

## 5. ĐỀ XUẤT CÔNG CỤ (RECOMMENDATIONS)

| Mục tiêu | Tool gợi ý | Lý do |
|---|---|---|
| Dễ bắt đầu | **NodeRAG** | Tích hợp sẵn logic GraphRAG, không cần cấu hình database phức tạp |
| Trực quan hóa tốt nhất | **Neo4j** | Giao diện đồ họa giúp "thấy" được tri thức đang được kết nối như thế nào |
| Nghiên cứu thuật toán | **NetworkX** | Cho phép can thiệp sâu vào các thuật toán toán học của đồ thị |

---

## 6. DELIVERABLES

Sinh viên nộp báo cáo bao gồm:

1. **Mã nguồn** (File `.py` hoặc `.ipynb`).
2. **Ảnh chụp màn hình** đồ thị tri thức đã xây dựng (từ Neo4j hoặc Matplotlib).
3. **Bảng so sánh** kết quả 20 câu hỏi benchmark giữa Flat RAG và GraphRAG.
4. **Phân tích ngắn gọn** về chi phí (Token usage, time) khi xây dựng đồ thị.

---

## 7. CHECKLIST HOÀN THÀNH

- [ ] Cài đặt môi trường thành công
- [ ] Extract được ít nhất 50 triples từ corpus
- [ ] Build đồ thị với NetworkX / Neo4j / NodeRAG
- [ ] Visualize đồ thị thành công
- [ ] Viết được hàm multi-hop query (2-hop)
- [ ] So sánh Flat RAG vs GraphRAG trên ≥ 5 câu hỏi
- [ ] Ghi nhận được ít nhất 1 trường hợp GraphRAG vượt trội Flat RAG
- [ ] Nộp báo cáo đủ 4 phần Deliverables

---

## 8. TÀI LIỆU THAM KHẢO

- [NodeRAG GitHub](https://github.com/Terry-Xu-666/NodeRAG)
- [Neo4j Python Driver Docs](https://neo4j.com/docs/python-manual/current/)
- [NetworkX Documentation](https://networkx.org/documentation/stable/)
- [Microsoft GraphRAG Paper](https://arxiv.org/abs/2404.16130)
- [LangChain Graph RAG Guide](https://python.langchain.com/docs/use_cases/graph/)
