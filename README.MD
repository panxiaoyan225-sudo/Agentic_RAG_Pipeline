## Full End-to-End Agentic RAG Pipeline  

My pipeline is an **end-to-end Agentic RAG system**.This is a simpler version built to demonstrate the work. The `RAG_pipeline_local_Model.py` executes the **full end-to-end workflow**: ingest, embed, retrieve, assemble a context-enriched prompt, and send it directly to a local **Ollama** inference endpoint (`llama3.1:latest` by default) to generate the final answer.    
 My deployed version uses **separate index and retrieval stages** with **incremental ingestion**. Start with **Langchain-RAG**, If you want rapid development, proof‑of‑concept, or to leverage the broader ecosystem. 

The agent has **planning, retrieval, and self-correction capabilities** — it decides how to query, deduplicates results, and scores whether the answer is grounded in the retrieved evidence. It implements a **priority cascade**: internal organizational context first, then live external knowledge with approval, then parametric fallback. It's not a single-shot retrieve-and-generate — the agent iterates over sub-queries and validates the answer against the source.  

## Priority cascade

A user question enters a fixed order. Internal evidence outranks the open web, and the open web outranks parametric memory.

```mermaid
flowchart LR
    Q["User query"] --> R1["1. Internal<br>organizational<br>context"]
    R1 --> R2["2. Live external<br>knowledge"]
    R2 --> R3["3. Pre-trained<br>parametric<br>knowledge"]
    R1 --> P["Context<br>enriched<br>prompt"]
    R2 --> P
    R3 --> P
    P --> LLM["Local Ollama<br>LLM endpoint<br>(llama3.1:latest)"]
```
1. **Internal organizational context — What the organization knows.** ChromaDB is searched for the nearest document chunks. Those passages ground the model in factual enterprise text, and each passage keeps the source file name and chunk index.
2. **Live External Knowledge — What the outside world knows now.** If the retrieved context is missing, incomplete, or the question needs live knowledge, the inference endpoint is instructed to use web search or other external sources only after approval.   
[Web Search: Why "Only with Approval" Is Critical ->](https://xiaoyanpan.com/Web_Search_Only_With_Approval.html)
3. **Pre-trained parametric knowledge — What the model generally knows.** General reasoning fills gaps that neither the index nor a live search covers.


 ## How the agent is Instructed & Governed  
 What the agent is authorized to do, what evidence it can rely on, and when it must stop.

```mermaid
flowchart TD
    %% Define Styles
    classDef anchor fill:#EDF2F7,stroke:#64748B,stroke-width:1.4px,color:#0F172A;
    classDef process fill:#E8F4FD,stroke:#BAE6FD,stroke-width:1.2px,color:#0C4A6E;
    classDef decision fill:#F5FAFF,stroke:#93C5FD,stroke-width:1.2px,stroke-dasharray: 4 3,color:#0C4A6E;
    classDef success fill:#E6F4EA,stroke:#4D8B6F,stroke-width:1.35px,color:#14532D;
    classDef warning fill:#FDF3E7,stroke:#C98A5B,stroke-width:1.35px,color:#7C2D12;

    %% Column 1: Entry & Internal Retrieval
    A["User Question"] --> B["Agent Planning"]
    B --> C["Internal Knowledge Retrieval"]
    C --> D["Evaluate Passages"]
    D --> E{"Sufficient?"}

    %% Column 2: Internal Sufficiency & Human Approval
    E -- "Yes" --> H["Assemble Context"]
    E -- "No" --> F["Live External Sources<br/>(Approval Required)"]
    F --> G{"Approval Granted?"}
    G -- "Yes" --> I["External Retrieval"]
    G -- "No" --> J["Not Authorized"]
    J --> M["Parametric Knowledge"]

    %% Column 3: External Retrieval & Parametric Fallback
    I --> K["Evaluate External"]
    K --> L{"Sufficient?"}
    L -- "Yes" --> H
    L -- "No" --> M
    M --> H

    %% Column 4: Context Assembly, Local Inference & Self-Correction
    H --> N["Local Ollama Inference"]
    N --> O["Self-Correction"]
    O --> P{"Grounded?"}

    %% Column 5: Governed Outcomes & Retry Loop
    P -- "Yes" --> Q["Return Answer"]
    P -- "No" --> R{"Refine Query?"}
    R -- "Yes" --> S["Retry Retrieval"]
    R -- "No" --> T["Stop & Escalate"]
    
    %% Retry Loop Back
    S --> C

    %% Apply Classes
    class A anchor;
    class B,C,D,F,I,K,J,M,N,O,S process;
    class E,G,L,P,R decision;
    class Q success;
    class T warning;
    class H anchor;
```

When retrieved evidence cannot support a conclusion, the run records that. `SimpleRAGAgent.self_correct()` scores lexical overlap between the generated answer and the retrieved passages and sets `supported` and `needs_rephrase`. A thin overlap means the agent leaned on fallback knowledge or needs a rephrased query rather than inventing a citation.

## Two strategies, and the hybrid that uses both

| | Strategy 1 — Retrieval-augmented generation | Strategy 2 — Enterprise fine-tuning |
|---|---|---|
| What it changes | The evidence in the prompt | The model's weights, tone, and terminology |
| What it is good at | Factual accuracy, fresh knowledge, source traceability | Behavior, persona, and house style |
| Strengths | Updates by dropping in new files. Access stays on the local index. Every passage names its source. | Deep customization. Low latency once the weights are static. |
| Limits | The answer is only as good as retrieval. The prompt has a context window. | Can still hallucinate. Training and every retrain are expensive. |

**The enterprise ideal is the hybrid.** Use RAG to pull up-to-date facts from documents. Use light fine-tuning only for tone, structure, and branding. This repository implements the RAG side: documents are chunked and embedded into ChromaDB without a training run. Adapting model weights is a separate job and is not part of the pipeline script.

## Why Internal Context Comes First

Organizations keep a local knowledge base because the facts change faster than a training cycle.

**Dynamic data, instant sync.** When a policy, price, or SOP changes, the updated file is dropped into the reports directory and the pipeline is run again. New chunks are added to ChromaDB in the same run. No GPU retraining is required for a knowledge update.

**Grounded answers with a source.** Retrieved passages are attached to the prompt as `[Source: filename]` blocks. The index stores the source file, chunk index, and extension with every vector, so an answer can be traced to the document it came from.

**Local-first knowledge base and inference.** PDFs, HTML, Word files, text, and Markdown land in one on-disk Chroma collection. Privacy follows the whole path: files stay in the reports directory you configure, vectors stay in the index directory you configure, and inference runs locally via Ollama (`http://localhost:11434`).

## How a run works

Running the script loads settings from `.env`, then walks five steps:

1. **Ingest.** `ingest_documents()` walks `RAG_REPORTS_DIR`, skips excluded directories and extensions, and reads HTML, PDF, DOCX, TXT, and Markdown.
2. **Chunk.** Each document is split into windows of `RAG_CHUNK_SIZE` characters with `RAG_CHUNK_OVERLAP` characters of overlap. A chunk id is `{stem}_chunk_{index}_{filename}`.
3. **Embed and store.** `embed_and_store()` loads `all-MiniLM-L6-v2`, encodes chunks in batches of 32, and `add`s them to the persistent Chroma collection. Existing ids are left in place.
4. **Retrieve.** The same embedding model encodes the question. Chroma returns the nearest `RAG_TOP_K` chunks with document text, metadata, and distance.
5. **Generate & Evaluate.** `SimpleRAGAgent` deduplicates the chunks, builds the system-and-user prompt following the 3-step priority cascade, sends it to the local Ollama model (`call_ollama()`), and scores answer-to-context overlap via `self_correct()`.

The interactive loop prints the plan, how many chunks came back, the overlap confidence, and the final **LLM Answer** generated by `llama3.1:latest`. Type `exit` to stop.

## Ingestion rules

| Rule | What the pipeline does |
|---|---|
| Skip rules | A path is skipped when any directory component is `.venv`, `.git`, `__pycache__`, `node_modules`, `.DS_Store`, or `.vs`. Shortcut files (`.lnk`, `.url`), bytecode (`.pyc`, `.pyo`, `.pyd`), binaries, images, and archives are skipped by extension. Matching is case-insensitive. |
| Multi-format ingestion | HTML, PDF, DOCX, TXT, and Markdown are read. HTML is stripped of script, style, and tags. PDF uses PyMuPDF. DOCX uses `python-docx` paragraph text. Markdown heading and emphasis markers are removed. **CSV is excluded** and is not in the allowed extension list. |
| Incremental updates | The Chroma collection is opened with `get_or_create_collection` on the existing on-disk index. `collection.add` inserts new chunk ids and does not delete what is already stored. A later run keeps existing chunks and adds chunks whose ids are not in the collection yet. |
| Embedding model | `all-MiniLM-L6-v2` through Sentence Transformers. It is an open-source model. Queries and documents are embedded with the same model so they occupy one vector space. Override the name with `RAG_EMBEDDING_MODEL` if you change it in both indexing and query. |
| DuplicateIDError handling | Built into `embed_and_store()`. If a batch `add` raises Chroma's `DuplicateIDError` (or an error whose message says the ids are duplicates), that batch is not discarded wholesale. The function reads the ids already in the collection, drops the ones that collide, and `add`s only the new ids. If every id in the batch already exists, the batch is skipped and the run continues. |

Chunk ids are derived from the file name and the chunk index. Re-running on an unchanged file is a no-op for those ids. If you edit a file but the chunk count and file name stay the same, the old vectors remain, because a duplicate id is skipped rather than overwritten. A renamed file or a newly added file produces new ids and is appended.

## Configuration

Paths and runtime settings live in `.env` next to the pipeline script. The Python file reads that file at startup and does not contain machine-specific directories. A variable already set in the process environment is left as-is.

| Variable | Required | Role |
|---|---|---|
| `RAG_REPORTS_DIR` | yes | Root folder walked for documents |
| `RAG_INDEX_DIR` | yes | Persistent ChromaDB directory |
| `RAG_CHUNKS_DIR` | yes | Directory created for chunk output |
| `RAG_EMBEDDING_MODEL` | no | Defaults to `all-MiniLM-L6-v2` |
| `RAG_COLLECTION_NAME` | no | Defaults to `rag_documents` |
| `RAG_CHUNK_SIZE` | no | Character window, default `500` |
| `RAG_CHUNK_OVERLAP` | no | Overlap between windows, default `100` |
| `RAG_TOP_K` | no | Passages retrieved per question, default `5` |
| `OLLAMA_MODEL` | no | Primary local Ollama model, defaults to `llama3.1:latest` |
| `OLLAMA_FALLBACK_MODEL` | no | Fallback Ollama model if primary fails/times out, defaults to `llama3.1:latest` |
| `OLLAMA_URL` | no | Local Ollama API endpoint, defaults to `http://localhost:11434` |

`.env` is listed in `.gitignore`. Keep the filled-in file on this machine.

If a required variable is missing, the process exits before it touches the index.

<div align="right">
  <blockquote>
    <p><strong>Technology enables retrieval.</strong></p>
    <p><strong>Organization provides context.</strong></p>
    <p><strong>Human determines how that knowledge should be used.</strong></p>
  </blockquote>
</div>