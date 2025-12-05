import pdfplumber
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import base64
import os
from dotenv import load_dotenv
from io import BytesIO

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-teacher")
model = SentenceTransformer("all-MiniLM-L6-v2")

file_path = "C:/Users/Harsh/Downloads/Modify_617.pdf"
file_name = os.path.basename(file_path)

# Load your PDF bytes (or file)
with open(file_path, "rb") as f:  # Put PDF here
    pdf_bytes = f.read()
text = ""
with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
    for page in pdf.pages[:5]:  # 5pg demo
        text += page.extract_text() or ""

chunks = [text[i:i+384] for i in range(0, len(text), 384)]
vectors = []
for i, chunk in enumerate(chunks):
    emb = model.encode(chunk).tolist()
    vectors.append({
    "id": f"{file_name}_{i}",
    "values": emb,
    "metadata": {"source": file_name, "text": chunk}})

index.upsert(vectors=vectors)
print(f"Pre-loaded {len(chunks)} demo chunks.")
