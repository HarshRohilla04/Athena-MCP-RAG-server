"""
1 change the file path
2 change the namespace_text and namespace_table
3 in tables change the metadata of book tag in both 
4 change the context chunk to name of the full book
"""



import pdfplumber
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import base64
import os
from dotenv import load_dotenv
from io import BytesIO
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("athena-rag-storage")
model = SentenceTransformer("all-MiniLM-L6-v2")

file_path = "C:/Users/Harsh/Downloads/modern_aproach_to_AI.pdf"
file_name = os.path.basename(file_path)

# 🚀 NAMESPACE SETUP - Clean organization!
NAMESPACE_TEXT = "ai-modern-approach-text"      
NAMESPACE_TABLES = "ai-modern-approach-tables"

def clean_text(text: str) -> str:
    """Clean textbook artifacts"""
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Page \d+', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'(\w+)-\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text.strip())
    text = re.sub(r'[©®™•※★]', '', text)
    return text.strip()

def table_to_markdown(table: list) -> str:
    """Convert table to markdown"""
    if not table or len(table) == 0:
        return ""
    try:
        df = pd.DataFrame(table[1:], columns=table[0])
        return df.to_markdown(index=False)
    except:
        return str(table)

print(f"📖 Processing FULL BOOK: {file_name}")

with open(file_path, "rb") as f:
    pdf_bytes = f.read()

full_text = ""
table_vectors = []
text_vectors = []

with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
    total_pages = len(pdf.pages)
    print(f"📄 Found {total_pages} total pages")
    
    for page_num, page in enumerate(pdf.pages, 1):
        print(f"Processing page {page_num}/{total_pages} ({page_num/total_pages*100:.1f}%)...")
        
        # Extract text + clean
        text = page.extract_text() or ""
        cleaned_text = clean_text(text)
        full_text += f"\n--- Page {page_num} ---\n{cleaned_text}\n"
        
        # ✅ TABLES → SEPARATE NAMESPACE
        tables = page.extract_tables()
        for table_idx, table in enumerate(tables):
            if table:
                md_table = table_to_markdown(table)
                if md_table.strip():
                    table_chunk = f"**Table {table_idx+1} (Page {page_num})**\n{md_table}"
                    emb = model.encode(table_chunk).tolist()
                    table_vectors.append({
                        "id": f"{file_name}_table_pg{page_num}_{table_idx}",
                        "values": emb,
                        "metadata": {
                            "source": file_name, 
                            "text": table_chunk,
                            "type": "table",
                            "page": page_num,
                            "book": "Artificial Intelligence: A Modern Approach"
                        }
                    })
        
        # 🆕 BATCH UPSERT TABLES EVERY 50 PAGES
        if page_num % 50 == 0 or page_num == total_pages:
            if table_vectors:
                batch_size = 50  # Smaller for tables
                for i in range(0, len(table_vectors), batch_size):
                    batch = table_vectors[i:i+batch_size]
                    index.upsert(vectors=batch, namespace=NAMESPACE_TABLES)
                    print(f"  📊 Upserted {len(batch)} TABLES → {NAMESPACE_TABLES}")
                table_vectors = []  # Reset

print("🧠 Semantic chunking full text...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100,
    separators=["\n\n", "\n## ", "\n### ", "\n", ". ", " "],
    length_function=len
)

chunks = splitter.split_text(full_text)
print(f"✂️ Creating {len(chunks)} text chunks...")

# ✅ TEXT → MAIN NAMESPACE
for i, chunk in enumerate(chunks):
    if len(chunk.strip()) < 50:
        continue
    
    context_chunk = f"📖 Artificial Intelligence: A Modern Approach (Russell & Norvig)\n📄 {file_name}\n\n{chunk}"
    emb = model.encode(context_chunk).tolist()
    
    text_vectors.append({
        "id": f"{file_name}_text_{i}",
        "values": emb,
        "metadata": {
            "source": file_name, 
            "text": chunk,
            "type": "text",
            "chunk_size": len(chunk),
            "book":  "Artificial Intelligence: A Modern Approach"
        }
    })

# Final TEXT batch upsert
batch_size = 100
for i in range(0, len(text_vectors), batch_size):
    batch = text_vectors[i:i+batch_size]
    index.upsert(vectors=batch, namespace=NAMESPACE_TEXT)
    print(f"  📄 Upserted {len(batch)} TEXT → {NAMESPACE_TEXT}")

# Final TABLES upsert (if any remain)
if table_vectors:
    batch_size = 50
    for i in range(0, len(table_vectors), batch_size):
        batch = table_vectors[i:i+batch_size]
        index.upsert(vectors=batch, namespace=NAMESPACE_TABLES)
        print(f"  📊 Final TABLES → {NAMESPACE_TABLES}")

print(f"\n🎉 FULL BOOK INGESTION COMPLETE!")
print(f"📊 {len(text_vectors)} TEXT vectors → '{NAMESPACE_TEXT}'")
print(f"📊 {len(table_vectors)} TABLE vectors → '{NAMESPACE_TABLES}'")

