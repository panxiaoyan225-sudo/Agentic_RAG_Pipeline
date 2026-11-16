#!/usr/bin/env python3
"""
Custom RAG Pipeline — ChromaDB + Custom Python (Open-Source Embedding)
=======================================================================

Overview
--------
A self-contained Retrieval-Augmented Generation (RAG) pipeline built with:

  * ChromaDB (PersistentClient)        — local vector database, persisted to disk.
  * sentence-transformers              — open-source embeddings (default: all-MiniLM-L6-v2).
  * Ollama (local LLM via HTTP API)    — e.g., llama3.1, deepseek-r1, qwen.

Supported document formats: HTML, PDF, DOCX, TXT, MD.
The pipeline automatically skips common code/build directories (.venv, .git,
__pycache__, node_modules, ...) and binary/media file extensions.

Five stages
-----------
1. Ingest & Chunk   — read documents from RAG_REPORTS_DIR, extract text,
                       split text into overlapping chunks.
2. Embed & Store    — embed chunks with SentenceTransformer and store them
                       in ChromaDB; DuplicateIDError is handled gracefully.
3. Retrieve         — embed the user query and search ChromaDB for the
                       top-k most similar chunks.
4. Agent            — SimpleRAGAgent builds a contextual prompt (retrieved
                       context + user question), calls the local Ollama model,
                       and runs a lightweight self-correction confidence check.
5. Main loop        — interactive REPL; type exit/quit/q to leave.

Configuration
-------------
All paths and tunables come from environment variables, loaded from a .env file
next to this script (or already present in the process environment). Existing
environment variables always win over .env values (os.environ.setdefault).

Required env vars:  RAG_INDEX_DIR, RAG_CHUNKS_DIR, RAG_REPORTS_DIR
Optional env vars (with defaults):  RAG_CHUNK_SIZE (500),
    RAG_CHUNK_OVERLAP (100), RAG_EMBEDDING_MODEL (all-MiniLM-L6-v2),
    RAG_COLLECTION_NAME (rag_documents), RAG_TOP_K (5),
    OLLAMA_MODEL (llama3.1:latest), OLLAMA_URL (http://localhost:11434),
    OLLAMA_FALLBACK_MODEL (llama3.1:latest).
"""
# This is a custom, dependency-light RAG implementation.
# A separate LangChain-based Ollama example lives in Langchain_RAG.py for
# rapid development; this file is self-contained.

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import List, Dict, Any

# ── Configuration ──────────────────────────────────────────────────────────
# Default .env location: a file named ".env" in the same folder as this script.
ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_dotenv(env_path: Path = ENV_PATH) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Existing environment variables take precedence, so we use
    os.environ.setdefault and never overwrite a variable already exported
    in the shell.

    Recognized .env syntax:
        KEY=VALUE
        KEY="VALUE"        (surrounding quotes are stripped)
        KEY='VALUE'
        export KEY=VALUE   (the leading 'export ' is removed)
        # comments and blank lines are ignored.

    Parameters
    ----------
    env_path : Path, optional
        Path to the .env file. Defaults to ENV_PATH. If the file does not
        exist, this function returns silently.
    """
    # Nothing to do if there is no .env file beside the script.
    if not env_path.is_file():
        return

    # Read the file and process it line by line.
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        # Skip blank lines and full-line comments.
        if not line or line.startswith("#"):
            continue

        # Strip a leading "export " so "export KEY=VALUE" parses like "KEY=VALUE".
        if line.startswith("export "):
            line = line[7:].strip()

        # Only lines containing "=" are valid assignments.
        if "=" not in line:
            continue

        # Split on the first "=" only, so a VALUE may itself contain "=".
        key, value = line.split("=", 1)
        key = key.strip()
        # Strip surrounding double or single quotes from the value.
        value = value.strip().strip('"').strip("'")

        if key:
            # setdefault keeps any variable already set in the environment.
            os.environ.setdefault(key, value)


def _require_env(name: str) -> str:
    """Return a required environment variable, or exit if it is missing.

    Used for paths the pipeline cannot function without (index dir,
    chunks dir, reports dir). Prints a clear error and terminates via
    SystemExit when the variable is unset or empty.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(
            f"Missing {name}. Set it in {ENV_PATH} or the process environment."
        )
    return value


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a fallback default.

    Returns ``default`` when the variable is unset or empty. Exits with an
    error message when the variable is set but is not a valid integer.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"{name} must be an integer, got {raw!r}")


def build_config() -> Dict[str, Any]:
    """Assemble the full pipeline configuration dictionary.

    Loads the .env file, then reads environment variables (with sensible
    defaults where appropriate) and returns a config dict consumed by every
    stage of the pipeline.

    Returns
    -------
    dict
        Configuration keys:
          chunk_size           - max characters per text chunk (default 500)
          chunk_overlap        - characters of overlap between chunks (default 100)
          embedding_model      - sentence-transformers model name
          collection_name      - ChromaDB collection name
          vector_db            - always "chroma" for this implementation
          index_dir            - REQUIRED: directory for the ChromaDB store
          chunks_dir           - REQUIRED: directory for chunk artifacts
          reports_dir          - REQUIRED: directory of source documents
          top_k                - number of chunks to retrieve per query (default 5)
          file_extensions      - supported source file extensions
          skip_dirs            - directory names to skip while walking
          skip_extensions      - file extensions to skip (binaries/media/etc.)

    Notes
    -----
    CSV is intentionally omitted because this pipeline targets prose
    documents, not tabular data.
    """
    load_dotenv()
    return {
        "chunk_size": _env_int("RAG_CHUNK_SIZE", 500),
        "chunk_overlap": _env_int("RAG_CHUNK_OVERLAP", 100),
        "embedding_model": os.environ.get("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
        or "all-MiniLM-L6-v2",
        "collection_name": os.environ.get("RAG_COLLECTION_NAME", "rag_documents").strip()
        or "rag_documents",
        "vector_db": "chroma",
        # These three directories are required; the pipeline aborts if missing.
        "index_dir": _require_env("RAG_INDEX_DIR"),
        "chunks_dir": _require_env("RAG_CHUNKS_DIR"),
        "reports_dir": _require_env("RAG_REPORTS_DIR"),
        "top_k": _env_int("RAG_TOP_K", 5),
        # CSV is intentionally omitted. Facts come from prose documents, not tables.
        "file_extensions": [".html", ".pdf", ".docx", ".txt", ".md"],
        "skip_dirs": [".venv", ".git", "__pycache__", "node_modules", ".DS_Store", ".vs"],
        "skip_extensions": [
            # Compiled bytecode / binaries / executables.
            ".pyc", ".pyo", ".pyd", ".dll", ".so", ".exe", ".bin",
            # Images.
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico",
            # Archives.
            ".zip", ".rar", ".7z", ".tar", ".gz",
            # Shortcuts / web links.
            ".lnk", ".url",
        ],
    }

# ── Helpers ────────────────────────────────────────────────────


def clean_html_text(html_content: str) -> str:
    """Strip HTML tags and scripts/styles, collapsing whitespace.

    Steps:
      1. Remove <script>...</script> blocks (case-insensitive, across lines).
      2. Remove <style>...</style> blocks (case-insensitive, across lines).
      3. Remove all remaining HTML tags (<...>).
      4. Collapse runs of whitespace into a single space and trim ends.

    Parameters
    ----------
    html_content : str
        Raw HTML source text.

    Returns
    -------
    str
        Cleaned plain-text content.
    """
    # Remove script blocks and everything inside them.
    text = re.sub(r'<script.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    # Remove style blocks and everything inside them.
    text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove every remaining HTML tag.
    text = re.sub(r'<[^>]+>', '', text)
    # Collapse any whitespace sequence (spaces, tabs, newlines) into one space.
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract plain text from a PDF file using PyMuPDF (fitz).

    Returns an empty string if fitz is not installed or if any error occurs,
    so the caller can decide how to handle an unreadable PDF.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        # Extract text from each page, then join with spaces.
        texts = [page.get_text("text") for page in doc]
        doc.close()
        return " ".join(texts)
    except ImportError:
        # fitz / PyMuPDF not available in this environment.
        return ""
    except Exception:
        # Any other read error (corrupt file, permissions, etc.).
        return ""


def extract_text_from_docx(docx_path: Path) -> str:
    """Extract plain text from a DOCX file using python-docx.

    Joins all paragraph texts with spaces. Returns an empty string if
    python-docx is missing or an error occurs.
    """
    try:
        from docx import Document
        doc = Document(docx_path)
        return " ".join([para.text for para in doc.paragraphs])
    except ImportError:
        return ""
    except Exception:
        return ""


def extract_text_from_txt(txt_path: Path) -> str:
    """Read a plain-text (.txt) file as UTF-8.

    Uses errors="replace" so undecodable bytes are replaced rather than
    raising an exception.
    """
    return txt_path.read_text(encoding="utf-8", errors="replace")


def extract_text_from_md(md_path: Path) -> str:
    """Read a Markdown file and lightly strip common Markdown formatting.

    Removes:
      - leading '# ' from headings (line by line)
      - **bold** markers, keeping the inner text
      - *italic* markers, keeping the inner text

    Returns the remaining plain text, stripped of leading/trailing whitespace.
    """
    text = md_path.read_text(encoding="utf-8", errors="replace")
    # Strip markdown heading markers ('# ' at the start of a line).
    text = re.sub(r'^# +', '', text, flags=re.MULTILINE)
    # Remove double-asterisk bold markers, keep the inner text.
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    # Remove single-asterisk italic markers, keep the inner text.
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split a long string into overlapping fixed-size chunks.

    Uses a sliding window: each chunk is ``chunk_size`` characters, and the
    next window starts ``overlap`` characters before the end of the current
    one, so no information is lost at the boundaries.

    Parameters
    ----------
    text : str
        The source text to split.
    chunk_size : int, optional
        Maximum number of characters per chunk (default 500).
    overlap : int, optional
        Number of overlapping characters between consecutive chunks (default 100).

    Returns
    -------
    list of str
        The resulting chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        # Move the window forward, leaving ``overlap`` characters shared.
        start = end - overlap
    return chunks


def should_skip_path(filepath: Path, config: dict) -> bool:
    """Decide whether a file path should be skipped during ingestion.

    A path is skipped if:
      * any directory component matches a name in config["skip_dirs"]
        (case-insensitive), e.g. .venv, .git, __pycache__; or
      * the file extension is in config["skip_extensions"],
        e.g. binaries, images, archives.

    Parameters
    ----------
    filepath : Path
        The file path to evaluate.
    config : dict
        Pipeline configuration containing "skip_dirs" and "skip_extensions".

    Returns
    -------
    bool
        True if the path should be skipped, False otherwise.
    """
    parts = filepath.parts
    # Check each path component against the skip-directory list (case-insensitive).
    for skip_dir in config["skip_dirs"]:
        if skip_dir.lower() in [p.lower() for p in parts]:
            return True

    ext = filepath.suffix.lower()
    # Skip files whose extension is on the skip-extensions list.
    if ext in config["skip_extensions"]:
        return True

    return False


# ── Step 1: Ingest & Chunk ─────────────────────────────────────
def ingest_documents(config: dict) -> List[Dict]:
    """Walk the reports directory, extract text from supported files, and chunk it.

    For every supported document found under config["reports_dir"]:
      1. Detect the file type by extension.
      2. Extract the raw text using the appropriate extractor.
      3. Skip empty / unreadable documents.
      4. Split the text into overlapping chunks.
      5. Assign each chunk a unique id, source filename, chunk index, and ext.

    Directories listed in config["skip_dirs"] and files listed in
    config["skip_extensions"] are ignored. Unsupported extensions are ignored.

    Parameters
    ----------
    config : dict
        Pipeline configuration.

    Returns
    -------
    list of dict
        Each dict has keys: id, text, source, chunk_index, ext.
    """
    reports_dir = Path(config["reports_dir"])
    if not reports_dir.exists():
        print(f"X Reports directory not found: {reports_dir}")
        return []

    all_chunks = []
    # Normalize supported extensions to lowercase for case-insensitive matching.
    supported_exts = [ext.lower() for ext in config["file_extensions"]]

    # os.walk yields (root, dirs, files) for every directory in the tree.
    for root, _, files in os.walk(reports_dir):
        root_path = Path(root)
        # Skip the entire directory if it matches a skip-directory name.
        if should_skip_path(root_path, config):
            continue

        for filename in files:
            filepath = root_path / filename
            ext = filepath.suffix.lower()

            # Ignore files whose extension is not in the supported list.
            if ext not in supported_exts:
                continue

            # Double-check the full path isn't otherwise skippable.
            if should_skip_path(filepath, config):
                continue

            print(f"[DOC] Ingesting: {filepath.relative_to(reports_dir)}")

            try:
                # Dispatch to the correct extractor based on file extension.
                if ext == ".html":
                    raw = filepath.read_text(encoding="utf-8")
                    text = clean_html_text(raw)
                elif ext == ".pdf":
                    text = extract_text_from_pdf(filepath)
                elif ext == ".docx":
                    text = extract_text_from_docx(filepath)
                elif ext == ".txt":
                    text = extract_text_from_txt(filepath)
                elif ext == ".md":
                    text = extract_text_from_md(filepath)
                else:
                    print(f"WARN Unsupported extension {ext}, skipping {filepath.name}")
                    continue

                # Skip documents that yielded no usable text.
                if not text or not text.strip():
                    print(f"WARN No text extracted from {filepath.name}")
                    continue

                # Split the extracted text into overlapping chunks.
                chunks = chunk_text(text, config["chunk_size"], config["chunk_overlap"])
                for idx, chunk_content in enumerate(chunks):
                    # Build a unique id: filename + chunk index + filename without ext.
                    all_chunks.append({
                        "id": f"{filepath.stem}_chunk_{idx}_{filename.replace(ext, '')}",
                        "text": chunk_content,
                        "source": filepath.name,
                        "chunk_index": idx,
                        "ext": ext,
                    })
            except Exception as e:
                # One bad file should not abort the whole ingestion run.
                print(f"X Error reading {filepath.name}: {e}")

    print(f"OK Created {len(all_chunks)} chunks from supported documents in {reports_dir}")
    return all_chunks


# ── Step 2: Embed & Store ──────────────────────────────────────
def embed_and_store(chunks: List[Dict], config: dict):
    """Embed text chunks and store them in a local ChromaDB collection.

    Steps:
      1. Open (or create) a persistent ChromaDB client at config["index_dir"].
      2. Get or create the collection named config["collection_name"].
      3. Load the SentenceTransformer embedding model.
      4. Process chunks in batches of 32:
           - embed the batch,
           - add documents/ids/embeddings/metadatas to ChromaDB,
           - on DuplicateIDError, filter out already-existing ids and retry.
      5. Return the collection object for later retrieval.

    Parameters
    ----------
    chunks : list of dict
        Output from ingest_documents(); each item has "id", "text", "source",
        "chunk_index", "ext".
    config : dict
        Pipeline configuration.

    Returns
    -------
    chromadb.Collection
        The ChromaDB collection containing the embedded chunks.
    """
    from chromadb import PersistentClient
    from chromadb.errors import DuplicateIDError
    from sentence_transformers import SentenceTransformer

    # PersistentClient saves the vector database to disk under index_dir.
    client = PersistentClient(path=config["index_dir"])
    # get_or_create_collection creates the collection if absent, otherwise opens it.
    collection = client.get_or_create_collection(config["collection_name"])

    print(f"EMBED Loading embedding model: {config['embedding_model']}")
    embedder = SentenceTransformer(config["embedding_model"])

    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        # Metadata stored alongside each vector for later citation/filtering.
        metadatas = [
            {"source": c["source"], "chunk_index": c["chunk_index"], "ext": c["ext"]}
            for c in batch
        ]

        # Convert texts to embedding vectors, then to Python lists for ChromaDB.
        embeddings = embedder.encode(texts, convert_to_numpy=True).tolist()

        try:
            collection.add(
                documents=texts,
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            print(f"OK Stored batch {i // batch_size + 1}: {len(batch)} chunks")
        except Exception as e:
            error_msg = str(e)
            # Detect whether the failure is due to duplicate ids already in ChromaDB.
            is_duplicate = (
                isinstance(e, DuplicateIDError)
                or "DuplicateIDError" in error_msg
                or "duplicate" in error_msg.lower()
            )
            if not is_duplicate:
                # Re-raise any non-duplicate error so it is not silently swallowed.
                raise

            print(f"WARN DuplicateIDError in batch {i // batch_size + 1}: {e}")
            print("   Skipping duplicate IDs, continuing...")
            try:
                # Fetch all ids currently present in the collection.
                existing_ids = set(collection.get(include=[]).get("ids", []))
                # Keep only ids not already stored.
                unique_ids = [cid for cid in ids if cid not in existing_ids]
                if unique_ids:
                    idx_map = {cid: idx for idx, cid in enumerate(ids)}
                    unique_texts = [texts[idx_map[cid]] for cid in unique_ids]
                    unique_metadatas = [metadatas[idx_map[cid]] for cid in unique_ids]
                    unique_embeddings = [embeddings[idx_map[cid]] for cid in unique_ids]
                    collection.add(
                        documents=unique_texts,
                        ids=unique_ids,
                        embeddings=unique_embeddings,
                        metadatas=unique_metadatas,
                    )
                    print(f"OK Retried batch: {len(unique_ids)} unique chunks stored")
                else:
                    print("   All IDs were duplicates, skipping batch.")
            except Exception as retry_err:
                print(f"X Retry failed: {retry_err}")

    print(f"OK Embedding complete. {len(chunks)} chunks stored at {config['index_dir']}")
    return collection


# ── Step 3: Retrieval ──────────────────────────────────────────
def retrieve(query: str, config: dict, collection) -> List[Dict]:
    """Retrieve the top-k chunks most similar to a query string.

    The query is embedded with the same SentenceTransformer model used at
    ingestion time, then ChromaDB performs an approximate nearest-neighbor
    search in the vector space.

    Parameters
    ----------
    query : str
        The user's question.
    config : dict
        Pipeline configuration (provides embedding_model and top_k).
    collection : chromadb.Collection
        The ChromaDB collection to search.

    Returns
    -------
    list of dict
        Each dict has keys: id, text, metadata, distance.
        ``distance`` is the ChromaDB distance metric (lower = more similar).
    """
    from sentence_transformers import SentenceTransformer

    # Reuse the exact same embedding model so query and document vectors are comparable.
    embedder = SentenceTransformer(config["embedding_model"])
    query_embedding = embedder.encode([query], convert_to_numpy=True)[0].tolist()

    # Ask ChromaDB for the top_k nearest vectors to the query embedding.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=config["top_k"],
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    print(f"SEARCH Retrieved {len(retrieved)} chunks for query: '{query[:50]}...'")
    return retrieved


# ── Step 4: Agent ──────────────────────────────────────────────
class SimpleRAGAgent:
    """A minimal agentic RAG wrapper around the local Ollama model.

    Responsibilities:
      * plan(query)        - decompose a question into sub-queries (currently a
                             single direct-retrieval sub-query).
      * retrieve_tool(...) - fetch relevant chunks from ChromaDB.
      * generate(...)      - build a system + user prompt from retrieved context.
      * call_ollama(...)   - send the prompt to the local Ollama HTTP API.
      * self_correct(...)  - estimate how well the answer is supported by context.
      * run(query)         - orchestrate plan -> retrieve -> generate -> answer
                             -> self-correct and return a result dict.
    """

    def __init__(self, config: dict, collection):
        self.config = config
        self.collection = collection
        # Ollama model to query (default llama3.1:latest).
        self.ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1:latest").strip() or "llama3.1:latest"
        # Base URL of the Ollama server, trailing slash removed.
        self.ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")

    def plan(self, query: str) -> Dict:
        """Create a retrieval plan for the user query.

        This simple implementation treats the whole question as one
        direct-retrieval sub-query. A more advanced agent could split the
        question into multiple sub-queries here.

        Returns
        -------
        dict
            {"query": <original>, "sub_queries": [<original>],
             "strategy": "direct_retrieval"}
        """
        return {"query": query, "sub_queries": [query], "strategy": "direct_retrieval"}

    def retrieve_tool(self, sub_query: str) -> List[Dict]:
        """Retrieve context chunks for a sub-query from ChromaDB."""
        return retrieve(sub_query, self.config, self.collection)

    def generate(self, query: str, contexts: List[Dict]) -> str:
        """Build the full prompt (system instructions + user question) for the LLM.

        The prompt foregrounds retrieved organizational context (with source
        citations) and instructs the model to prioritize that context, fall
        back to web search / real-time data when context is incomplete, and
        finally use general knowledge. It also tells the model never to refuse
        an answer merely because the retrieved context is thin.

        Parameters
        ----------
        query : str
            The user's original question.
        contexts : list of dict
            Retrieved chunks (each with "metadata" containing "source" and
            "text").

        Returns
        -------
        str
            The complete prompt string ready to send to the LLM.
        """
        # Build a context block, one section per retrieved chunk.
        if contexts:
            context_lines = []
            for ctx in contexts:
                source = ctx.get("metadata", {}).get("source", "unknown")
                text = ctx.get("text", "")
                context_lines.append(f"[Source: {source}]\n{text}\n")
            context_documents = "\n---\n".join(context_lines)
        else:
            context_documents = ""

        if context_documents.strip():
            context_block = f"### RETRIEVED ORGANIZATIONAL CONTEXT:\n{context_documents.strip()}"
        else:
            context_block = "### RETRIEVED ORGANIZATIONAL CONTEXT:\nNo relevant internal organizational context found."

        user_prompt = f"""{context_block}
### USER QUESTION: {query}
### ANSWER:"""

        system_prompt = """You are an expert AI assistant equipped with access to internal organizational context and web search capability. Your primary objective is to provide the most accurate, up-to-date, and helpful answer by following this priority order:
1. Organizational Context (Primary): Use the provided retrieved context to answer questions specific to internal workflows, documents, and data. Always cite the file source when referencing internal information.
2. Web Search & Real-Time Data (Secondary): If the retrieved context is missing, incomplete, or requires current real-time facts (e.g., current dates, dynamic market stats, recent technical updates), explicitly use your online search capabilities to find up-to-date answers.
3. General Knowledge (Fallback): Use your internal pre-trained reasoning and knowledge to synthesize a complete answer if context is absent or supplementary details are needed.
Formatting Rules:
- Clearly state when you are drawing from internal context versus online search or general knowledge.
- Never refuse to answer simply because information is missing from the retrieved context.
- Keep answers precise, well-structured, and accurate."""

        return f"{system_prompt}\n\n{user_prompt}"

    def self_correct(self, query: str, answer: str, contexts: List[Dict]) -> Dict:
        """Estimate how well the generated answer is supported by retrieved context.

        Computes a crude confidence score as the fraction of answer words that
        also appear in the context text (case-insensitive). This is a heuristic,
        not a rigorous factuality check.

        Returns
        -------
        dict
            {
              "confidence": float,   # 0.0 to 1.0
              "supported": bool,     # True if confidence > 0.4
              "needs_rephrase": bool # True if confidence < 0.3
            }
        """
        context_text = " ".join([c["text"] for c in contexts]).lower()
        answer_words = set(answer.lower().split())
        context_words = set(context_text.split())
        overlap = len(answer_words & context_words)
        confidence = min(1.0, overlap / max(len(answer_words), 1))
        return {
            "confidence": round(confidence, 2),
            "supported": confidence > 0.4,
            "needs_rephrase": confidence < 0.3,
        }

    def call_ollama(self, prompt: str) -> str:
        """Send a prompt to the local Ollama model via its HTTP /api/generate endpoint.

        Tries the primary model (self.ollama_model) up to two times; if both
        attempts fail, falls back to OLLAMA_FALLBACK_MODEL (default
        llama3.1:latest), also up to two times. Returns the model's response
        text, or an error string if every attempt fails.

        Parameters
        ----------
        prompt : str
            The full prompt generated by self.generate().

        Returns
        -------
        str
            The model's response text, or "[Ollama error: ...]".
        """
        primary = self.ollama_model
        fallback = os.environ.get("OLLAMA_FALLBACK_MODEL", "llama3.1:latest").strip() or "llama3.1:latest"

        # Non-streaming request body; we read the complete response at once.
        data = json.dumps({"prompt": prompt, "stream": False}).encode("utf-8")
        last_err = None

        for model in [primary, fallback]:
            payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
            for attempt in range(2):
                try:
                    req = urllib.request.Request(
                        f"{self.ollama_url}/api/generate",
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    # 300-second timeout gives large models time to answer.
                    with urllib.request.urlopen(req, timeout=300) as resp:
                        body = json.loads(resp.read().decode("utf-8"))
                        return body.get("response", "").strip()
                except Exception as e:
                    last_err = e

        return f"[Ollama error: {last_err}]"

    def run(self, query: str) -> Dict:
        """Execute the full agentic RAG flow for a single user query.

        1. plan(...)        -> sub-queries
        2. retrieve_tool(...) for each sub-query -> raw chunks
        3. Deduplicate chunks by id and cap at config["top_k"]
        4. generate(...)    -> build prompt
        5. call_ollama(...) -> LLM answer
        6. self_correct(...) -> confidence / support heuristics

        Returns
        -------
        dict
            {
              "query": str,
              "answer": str,
              "plan": dict,
              "retrieved_contexts": list of dict,
              "prompt": str,
              "confidence": float,
              "supported": bool,
              "needs_rephrase": bool,
            }
        """
        plan = self.plan(query)
        all_contexts = []
        for sub_q in plan["sub_queries"]:
            ctxs = self.retrieve_tool(sub_q)
            all_contexts.extend(ctxs)

        # Remove duplicate chunks (same id) while preserving order.
        seen = set()
        unique_ctxs = []
        for c in all_contexts:
            if c["id"] not in seen:
                seen.add(c["id"])
                unique_ctxs.append(c)
        # Keep only the top_k most relevant contexts.
        unique_ctxs = unique_ctxs[: self.config["top_k"]]

        generation = self.generate(query, unique_ctxs)
        answer = self.call_ollama(generation)
        correction = self.self_correct(query, answer, unique_ctxs)

        return {
            "query": query,
            "answer": answer,
            "plan": plan,
            "retrieved_contexts": unique_ctxs,
            "prompt": generation,
            "confidence": correction["confidence"],
            "supported": correction["supported"],
            "needs_rephrase": correction["needs_rephrase"],
        }


# ── Main Pipeline ──────────────────────────────────────────────
def main():
    """Entry point for the interactive RAG pipeline.

    Runs the five stages:
      1. Build configuration from environment / .env.
      2. Create index_dir and chunks_dir if missing.
      3. Ingest & chunk documents from Reports/.
      4. Embed & store chunks in ChromaDB.
      5. Start an interactive Q&A loop with the agent until the user exits.
    """
    print("=" * 60)
    print("ROCKET RAG Pipeline - ChromaDB + Custom Python")
    print("=" * 60)

    config = build_config()

    # Ensure the directories for the vector store and chunk artifacts exist.
    Path(config["index_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["chunks_dir"]).mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Ingesting documents from Reports/ ...")
    chunks = ingest_documents(config)
    if not chunks:
        print("X No documents found. Add supported files to Reports/ and re-run.")
        return

    print("\n[2/5] Embedding & storing in ChromaDB ...")
    collection = embed_and_store(chunks, config)

    print("\n[3/5] Initializing agentic RAG agent ...")
    agent = SimpleRAGAgent(config, collection)

    print("\n[4/5] Agent ready. Type 'exit' to quit.\n")
    while True:
        try:
            query = input("SEARCH Your question: ").strip()
            if query.lower() in ("exit", "quit", "q"):
                break
            if not query:
                continue

            result = agent.run(query)

            print(f"\nCHART Plan: {result['plan']['strategy']}")
            print(f"DOWNLOAD Retrieved {len(result['retrieved_contexts'])} chunks")
            print(f"TARGET Confidence: {result['confidence']}")
            print(f"OK Supported: {result['supported']}")
            print(f"EMBED Needs rephrase: {result['needs_rephrase']}")
            if "answer" in result:
                print(f"\n🤖 LLM ANSWER:\n{result['answer']}\n")
            else:
                print("\n⚠ No LLM answer returned (call_ollama may have failed).")

            print(f"\n{'=' * 60}\n")

        except KeyboardInterrupt:
            # Ctrl+C exits the loop cleanly.
            break

    print("\nOK Pipeline complete. Goodbye!")


if __name__ == "__main__":
    main()