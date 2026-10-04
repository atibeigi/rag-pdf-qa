# PDF Question Answering with RAG (Gemini + FAISS)

A Retrieval-Augmented Generation (RAG) system that answers questions about a PDF document.
The document used is the paper **"AI-Generated Versus Human Text: Introducing a New Dataset for Benchmarking and Analysis"** (IEEE Transactions on Artificial Intelligence, 2025).

## How it works

```
PDF → text extraction (PyMuPDF) → remove references → chunking
    → Gemini embeddings → FAISS index
Question → query embedding → top-k retrieval → prompt → Gemini LLM → answer + sources
```

1. **Extraction:** PyMuPDF extracts text page by page, so every chunk keeps its page number.
2. **Cleaning:** the reference list is removed to reduce retrieval noise.
3. **Chunking:** ~800 characters with 150 characters of overlap, cut at word boundaries.
4. **Embedding:** `gemini-embedding-001` (768 dimensions), using `RETRIEVAL_DOCUMENT` for chunks and `RETRIEVAL_QUERY` for questions. Vectors are L2-normalized.
5. **Vector store:** FAISS `IndexFlatIP` (inner product = cosine similarity on normalized vectors).
6. **Retrieval:** top-6 chunks per question.
7. **Generation:** `gemini-3.6-flash` (temperature 0.2) with a prompt that restricts the answer to the retrieved context, requires page citations, and returns a fixed refusal sentence when the answer is not in the document.

## Design choices

| Choice | Reason |
|---|---|
| Chunk size 800 / overlap 150 | [your reasoning, e.g. large enough to keep a full idea, small enough to keep retrieval precise] |
| Gemini embeddings | [e.g. higher similarity scores than all-MiniLM-L6-v2 in my tests (0.66–0.82 vs 0.2–0.4); same ecosystem as the LLM] |
| FAISS | Simple, fast, no server needed; the document is small |
| top_k = 6 | Broad questions need several scattered chunks; 4 was too few |
| Strict prompt | Reduces hallucination; handles out-of-scope questions |

## Setup and run

1. Open `Task_QA.ipynb` in Google Colab.
2. Upload `paper.pdf`.
3. Run all cells (`Runtime → Run all`).


## Results

The system was tested with 10 questions, including one out-of-scope question
("Who won the FIFA World Cup in 2018?"), which was correctly refused.

- `rag_results.txt`: full output (answers + retrieved sources with similarity scores)
- `evaluation_table.csv`: manual comparison of each answer with the paper

Summary: [X] of 10 answers fully correct, [Y] partially correct, [Z] incorrect.

## Possible improvements

- Reranking (cross-encoder) and hybrid search (BM25 + embeddings)
- Section-aware chunking and better table extraction
- A similarity threshold to reject out-of-scope questions before calling the LLM
- Automated evaluation (e.g. LLM-as-judge or RAGAS)

## Files

| File | Description |
|---|---|
| `Task_Q&A.ipynb` | Full implementation |
| `paper.pdf` | Source document |
| `rag_results.txt` | Raw output of the test questions |
| `evaluation_table.csv` | Manual evaluation table |
