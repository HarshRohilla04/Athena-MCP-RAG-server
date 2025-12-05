# Athena MCP RAG Server

Production-ready **Pinecone RAG Server** for Claude Desktop with PDF ingestion and AI teaching tools.

---

##  Quick Setup

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/HarshRohilla04/Athena-MCP-RAG-server.git
cd Athena-MCP-RAG-server
uv sync
```

```bash
uv pip install -r requirements.txt
# OR
uv sync  # uses pyproject.toml
```
### 2. Pinecone Setup
-Create account at [https://www.pinecone.io/](https://www.pinecone.io/)

-Go to API Keys → copy your api key

-Create index:
```bash
Name: rag-storage   # name your rag storage 
Modality: Text  
Vector type: Dense  
Dimension: 384  
Metric: cosine 
```
### 3. Create .env
```bash
PINECONE_API_KEY = "your api key"
```
### 4. Main.py Changes
```bash
# ----------------------------------------------------------------------
# Global Configuration (initialized once at startup)
# ----------------------------------------------------------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("name of the index")
```
### 5. Install to Claude Desktop
```bash
uv run fastmcp install claude-desktop main.py
```
### 6. Configure Claude Desktop
```bash
{
  "SERVER_NAME": {
    "command": "C:/Users/YOUR_USERNAME/Athena-MCP-RAG-server/.venv/Scripts/python.exe",
    "args": ["C:/Users/YOUR_USERNAME/Athena-MCP-RAG-server/main.py"],
    "env": {
      "PINECONE_API_KEY": "your api key"
    },
    "transport": "stdio",
    "cwd": "C:/Users/YOUR_USERNAME/Athena-MCP-RAG-server",
    "timeout": 600
  }
}
```
### 7. Launch
1. Save the JSON file

2. Close Claude Desktop completely

3. Open Task Manager → end all Claude processes

4. Restart Claude Desktop

## ATHENA is ready for use!
