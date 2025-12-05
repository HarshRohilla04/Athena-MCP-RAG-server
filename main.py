# main_copy.py - Mini Pinecone RAG Teacher MCP
from fastmcp import FastMCP, Context
import base64
import pdfplumber
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
import json
from dotenv import load_dotenv
import time
import re

# ----------------------------------------------------------------------
# Load environment variables (PINECONE_API_KEY required)
# ----------------------------------------------------------------------
load_dotenv()

# ----------------------------------------------------------------------
# Athena — Your Personal AI Professor
# ----------------------------------------------------------------------
mcp = FastMCP(
    name="Athena",
    description="Your personal AI professor that reads your research papers and teaches you",
    version="1.0.0"
)

# ----------------------------------------------------------------------
# Global Configuration (initialized once at startup)
# ----------------------------------------------------------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME", "athena-rag-storage")
index = pc.Index(index_name)

# Embedding model - all-MiniLM-L6-v2 is fast and effective for RAG
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ----------------------------------------------------------------------
# TOOLS
# ----------------------------------------------------------------------
@mcp.tool()
def list_documents() -> str:
    """List all documents currently in the knowledge base."""
    try:
        broad_emb = model.encode("any document").tolist()
        results = index.query(vector=broad_emb, top_k=1000, include_metadata=True)
        sources = {}
        for m in results["matches"]:
            src = m["metadata"].get("source", "unknown.pdf")
            sources[src] = sources.get(src, 0) + 1
        
        docs = sorted([{"name": name, "chunks": count} for name, count in sources.items()], 
                    key=lambda x: x["name"])
        
        if not docs:
            return "No documents found in the knowledge base yet. Upload one using ingest_pdf!"
        
        summary = f"Found {len(docs)} document(s):\n\n"
        for d in docs:
            summary += f"• {d['name']} ({d['chunks']} chunks)\n"
        
        return summary + "\nUse summarize_document with one of these names."
    except Exception as e:
        return f"Error listing documents: {str(e)}"


@mcp.tool()
def knowledge_base_stats() -> str:
    """Show current knowledge base  health."""
    try:
        stats = index.describe_index_stats()
        return f"""Knowledge Base Status:
• Total vectors: {stats.get('total_vector_count', 0):,}
• Dimensions: {stats.get('dimension', 384)}
• Index fullness: {stats.get('index_fullness', 0.0):.3f}
• Namespaces: {list(stats.get('namespaces', {}).keys()) or ['default']}"""
    except Exception as e:
        return f"Error getting stats: {str(e)}"


@mcp.tool()
def summarize_document(document_name: str) -> str:
    """
    Summarize a specific document. 
    If the name is wrong or missing, I'll help you find the right one.
    """
    try:
        # First: List documents to get valid names
        broad_emb = model.encode("any document").tolist()
        results = index.query(vector=broad_emb, top_k=1000, include_metadata=True)
        valid_names = {m["metadata"].get("source", "") for m in results["matches"]}
        
        if not valid_names:
            return "No documents in the knowledge base yet! Upload one first."

        # Clean input
        name = document_name.strip().strip('"\'')
        
        if name.lower() in ["help", "list", "what", "?"]:
            return f"""Available documents:\n""" + "\n".join(f"• {n}" for n in sorted(valid_names)) + \
                f"\n\nReply with: summarize_document <name>"

        if name not in valid_names:
            closest = max(valid_names, key=lambda x: len(set(x.lower().split()) & set(name.lower().split())), default=None)
            suggestion = f" Did you mean '{closest}'?" if closest else ""
            return f"Document '{name}' not found.{suggestion}\n\nAvailable documents:\n" + \
                "\n".join(f"• {n}" for n in sorted(valid_names))

        # Actual summary
        query_emb = model.encode(f"summary of {name}").tolist()
        results = index.query(
            vector=query_emb,
            top_k=12,
            filter={"source": {"$eq": name}},
            include_metadata=True
        )
        
        chunks = [m["metadata"]["text"][:500] for m in results["matches"]]
        full = "\n\n".join(chunks).strip()
        
        if len(full) > 3000:
            full = full[:3000] + "\n\n[...truncated...]"
            
        return f"Summary of **{name}**:\n\n{full}"
        
    except Exception as e:
        return f"Error summarizing document: {str(e)}"
    
@mcp.tool()
def add_document(file_content: str, file_name: str) -> str:
    """Upload and index a PDF document into your private knowledge base."""
    try:
        start_time = time.time()
        
        
        pdf_bytes = base64.b64decode(
            file_content.split(",")[1] if "," in file_content else file_content
        )
        
        
        text = ""
        with pdfplumber.open(pdf_bytes) as pdf:
            total_pages = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if not text.strip():
            return "No text could be extracted from the PDF."
        
        
        chunk_size = 512
        overlap = 100
        chunks = [
            text[i:i + chunk_size]
            for i in range(0, len(text), chunk_size - overlap)
            if len(text[i:i + chunk_size].strip()) > 50
        ]
        
        if not chunks:
            return "No meaningful text chunks created."
        
        
        batch_size = 100
        vectors = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embeddings = model.encode(batch, batch_size=32, show_progress_bar=False)
            for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                vectors.append({
                    "id": f"{file_name}_{i + j}",
                    "values": embedding.tolist(),
                    "metadata": {
                        "source": file_name,
                        "text": clean_text(chunk),
                        "chunk_index": i + j
                    }
                })
        
        
        for i in range(0, len(vectors), 1000):
            index.upsert(vectors=vectors[i:i + 1000])
        
        elapsed = time.time() - start_time
        return f"Successfully added '{file_name}'\n" \
            f"• {len(chunks)} chunks created\n" \
            f"• {total_pages} pages processed\n" \
            f"• Took {elapsed:.1f}s"
    except Exception as e:
        return f"Failed to add document: {str(e)}"

@mcp.tool()
def ask_teacher(question: str, top_k: int = 5) -> str:
    """Ask using Pinecone RAG."""
    try:
        query_emb = model.encode(question).tolist()
        results = index.query(
            vector=query_emb, top_k=top_k, include_metadata=True
        )
        context = "\n".join(
            [m["metadata"]["text"][:200] + "..." for m in results["matches"]]
        )
        return f"RAG context ({top_k} chunks): {context}\nAnswer: [Simulated LLM response to '{question}']"
    except Exception as e:
        return f"Error: {str(e)}"
    
@mcp.tool()
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search your private documents and return the most relevant excerpts."""
    try:
        emb = model.encode(query).tolist()
        results = index.query(vector=emb, top_k=top_k, include_metadata=True)
        excerpts = []
        for i, m in enumerate(results["matches"], 1):
            text = m["metadata"]["text"][:400]
            excerpts.append(f"{i}. [{m['metadata']['source']}] {text}...")
        return "\n\n".join(excerpts) if excerpts else "No relevant content found."
    except Exception as e:
        return f"Search error: {str(e)}"
    
@mcp.tool()
def remove_document(document_name: str, confirm: str = "") -> str:
    """
    Permanently delete a specific document from your knowledge base.
    Requires exact name + confirmation to prevent accidents.
    """
    try:
        # Get current list
        broad_emb = model.encode("any").tolist()
        results = index.query(vector=broad_emb, top_k=1000, include_metadata=True)
        valid_names = {m["metadata"].get("source") for m in results["matches"] if m["metadata"].get("source")}
        
        name = document_name.strip().strip('"\'')
        
        if name not in valid_names:
            return f"Document '{name}' not found.\n\nAvailable documents:\n" + \
                "\n".join(f"• {n}" for n in sorted(valid_names))
        
        if confirm != f"DELETE {name}":
            return f"Are you sure you want to permanently delete '{name}'?\n\n" + \
                f"Type: remove_document \"{name}\" \"DELETE {name}\""
        
        # Actually delete from Pinecone
        emb = model.encode(f"placeholder for {name}").tolist()
        delete_results = index.query(vector=emb, top_k=1000, include_metadata=True)
        ids_to_delete = [
            m["id"] for m in delete_results["matches"] 
            if m["metadata"].get("source") == name
        ]
        
        if ids_to_delete:
            index.delete(ids=ids_to_delete)
            return f"Successfully deleted '{name}' ({len(ids_to_delete)} vectors removed)."
        else:
            return f"Found document but no vectors to delete (already gone?)."
            
    except Exception as e:
        return f"Error removing document: {str(e)}"


@mcp.tool()
def clear_knowledge_base(confirm: str) -> str:
    """Delete ALL documents (use only if you type 'YES DELETE EVERYTHING')"""
    if confirm != "YES DELETE EVERYTHING":
        return "Type exactly: clear_knowledge_base YES DELETE EVERYTHING"
    # Warning: Only enable if you add Pinecone delete logic
    return "Knowledge base cleared! (Not implemented yet — safe mode)"


# ----------------------------------------------------------------------
# PROMPTS (unchanged – you can keep the kb:// references)
# ----------------------------------------------------------------------

@mcp.prompt()
def weekly_progress_report() -> str:
    """Beautiful weekly summary of my learning journey."""
    return """Generate my Weekly AI Learning Report

Include:
• Documents added this week
• Total knowledge base size
• Top 3 insights I’ve internalized
• My growth trajectory
• One area to focus on next week
• Motivational quote from my documents (or make one up)

Format like a beautiful newsletter."""

@mcp.prompt()
def compare_concepts(concept1: str, concept2: str) -> str:
    """Deep comparison of two concepts using my private documents."""
    return f"""Compare '{concept1}' vs '{concept2}' using content from my knowledge base.

Structure:
1. One-sentence definition of each
2. Key similarities
3. Key differences (be precise!)
4. Real examples from my documents
5. Which one to use when?
6. Quick decision flowchart

Use ask tool on both concepts first."""

@mcp.prompt()
def generate_study_plan(
    topic: str, days: int = 7, level: str = "beginner"
) -> str:
    return f"""You are my personal learning coach. Create optimal {days}-day plan for '{topic}' ({level}).

First: Call list_knowledge_base and summarize_document on the most relevant paper.

Then create a {days}-day study plan for {topic} at {level} level.

Each day:
• Goal
• 2–3 key concepts from my documents
• One hands-on exercise
• 5-minute quiz question

Time: 60–90 mins/day. End with motivation."""


@mcp.prompt()
def generate_quiz(topic: str, num_questions: int = 6, difficulty: str = "medium") -> str:
    """Generate a quiz on {topic} using my private knowledge base."""
    return f"""Create a {difficulty} quiz on '{topic}' with {num_questions} questions (4 MCQ, 2 short answer).

First: Use list_knowledge_base and ask tool to pull real content from my documents.

Include:
• Question
• Options (A-D)
• Correct answer
• Detailed explanation using actual paper content

Make it educational and fun."""


@mcp.prompt()
def explain_concept(
    concept: str,
    level: str = "beginner",
    context: str = "",
    examples: bool = True,
) -> str:
    return f"""You are an exceptional teacher with access to my private documents, explaining '{concept}'.

Level: {{level}} (beginner: simple analogies; intermediate: math; advanced: research).

RAG context: {{context}}

Steps:
1. First check knowledge_base_stats and list_knowledge_base
2. Use ask tool to pull relevant info from my documents
3. Explain clearly with examples
4. End with one practice question

Structure:
1. Definition + analogy
2. How it works (steps/code)
3. Examples: {{'yes' if examples else 'no'}}
4. Applications (e.g., chest X-ray)
5. Pros/cons, alternatives
6. Practice question

Be encouraging, precise, Clear, engaging, precise."""


# ----------------------------------------------------------------------
# RESOURCES – **use the globals directly**
# ----------------------------------------------------------------------
@mcp.resource("kb://documents/list/{ignored}")
def documents_list(ignored: str = "list", ctx: Context | None = None) -> str:
    """List all ingested documents with chunk counts."""
    try:
        # broad query – any vector will return *all* vectors because the index is small
        broad_emb = model.encode("any document").tolist()
        results = index.query(
            vector=broad_emb, top_k=1000, include_metadata=True
        )

        sources: dict[str, int] = {}
        for m in results["matches"]:
            src = m["metadata"].get("source", "unknown.pdf")
            sources[src] = sources.get(src, 0) + 1

        docs = [
            {"name": name, "chunks": count}
            for name, count in sorted(sources.items())
        ]
        return json.dumps(
            {"documents": docs, "total_documents": len(docs)}, indent=2
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("kb://pinecone/stats/{ignored}")
def pinecone_stats(ignored: str = "stats", ctx: Context | None = None) -> str:
    """Pinecone index statistics."""
    try:
        stats = index.describe_index_stats()
        return json.dumps(
            {
                "total_vectors": stats.get("total_vector_count", 0),
                "dimensions": stats.get("dimension", 384),
                "index_fullness": stats.get("index_fullness", 0.0),
                "namespaces": list(stats.get("namespaces", {}).keys()),
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("kb://documents/{doc_name}/summary")
def get_doc_summary(doc_name: str, ctx: Context | None = None) -> str:
    """Summarize a specific document by name."""
    try:
        query_emb = model.encode(f"summary of {doc_name}").tolist()
        results = index.query(
            vector=query_emb,
            top_k=8,
            filter={"source": {"$eq": doc_name}},
            include_metadata=True,
        )

        chunks = [m["metadata"]["text"][:300] for m in results["matches"]]
        summary = "\n\n".join(chunks).strip()
        if not summary:
            summary = f"No chunks found for document: {doc_name}"

        return json.dumps(
            {"document": doc_name, "summary": summary}, indent=2
        )
    except Exception as e:
        return json.dumps({"error": str(e)})
    
    


# ----------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()