# FEMA 306 Structural Diagnosis Agent

This project implements an AI-driven agent for diagnosing failure modes in unreinforced masonry (URM) structures according to FEMA 306 standards. It utilizes a Retrieval-Augmented Generation (RAG) pipeline empowered by a Hybrid Confidence Scoring mechanism to provide robust and explainable diagnoses.

## Key Features

*   **FEMA 306 Compliance**: Strictly adheres to the classification guides for URM piers and spandrels.
*   **Hybrid RAG Pipeline**: Combines semantic search with rule-based filtering.
*   **Hybrid Confidence Scoring**: Quantifies diagnosis reliability using a weighted combination of three distinct metrics.
*   **Explainable AI**: Provides detailed reasoning ("Scoped Reasoning") and exact citations for every diagnosis.

## Hybrid Confidence Score (Paper-Ready Method)

To ensure scientific rigor and reliability, we implement a **Hybrid Confidence Score** that triangulates correctness from three sources. This corresponds to the methodology described in our research:

**Formula:**
$$ \text{Score}_{\text{final}} = 0.5 \times \text{LLM}_{\text{SC}} + 0.3 \times \text{RC} + 0.2 \times \text{ReC} $$

### 1. LLM Self-Confidence ($\text{LLM}_{\text{SC}}$)
*   **Source**: The Large Language Model (Gemini 2.5).
*   **Method**: The model is prompted to self-evaluate its selection and assign a confidence score (0-1) based on semantic completeness and rule alignment.
*   **Weight**: 50% (Primary decision maker).

### 2. Retrieval Confidence ($\text{RC}$)
*   **Source**: Embedding Space (Sentence Transformers).
*   **Method**: We compute the Softmax over the similarity scores of the top-$k$ retrieved chunks. This measures how "distinct" the top result is compared to other candidates in the vector space.
    $$ p_i = \frac{e^{s_i}}{\sum e^{s_j}} $$
*   **Weight**: 30% (Data grounding).

### 3. Relevance Confidence ($\text{ReC}$)
*   **Source**: Cross-Encoder (`ms-marco-MiniLM-L6-v2`).
*   **Method**: A specialized BERT-based cross-encoder re-ranks the (Query, Document) pairs. It provides a more accurate semantic relevance signal than bi-encoder embeddings alone. Scores are min-max normalized.
*   **Weight**: 20% (Verification layer).

## Directory Structure

```
.
├── data/
│   ├── text_chunks/       # JSON chunks of FEMA 306 sections
│   └── chroma_db/         # Vector database (ChromaDB)
├── scripts/
│   ├── tag_failure_modes.py # Utility to tag metadata
│   └── test_agent.py      # Main CLI for testing diagnoses
├── src/
│   ├── structural_agent.py # Core logic (Hybrid Scoring implemented here)
│   ├── retriever.py       # Vector retrieval logic
│   ├── llm_service.py     # Gemini LLM interface
│   └── schema.py          # Pydantic data models
└── README.md              # This file
```

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install langchain chromadb sentence-transformers numpy
    ```

2.  **Run the Test Agent**:
    ```bash
    python3 scripts/test_agent.py
    ```

## Example Output

```text
Diagnosis: URM Weaker Pier Diagonal Tension Classification Guide
FAILURE MODE: TENSION
Confidence: 1.0
HYBRID SCORE: 0.9421
  > LLM Confidence: 1.0
  > Retrieval Conf (RC): 0.1321
  > Cross-Encoder Conf (ReC): 1.0000
Damage Level: Moderate (Moderate)
```
