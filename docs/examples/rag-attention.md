# Real-Run RAG Evaluation: "Attention Is All You Need" (Vaswani et al., 2017)

**Document Ingested**: `attention.pdf` (arXiv:1706.03762v7, 2.1 MB)  
**Ingestion Stats**: 1 document, 91 chunks, SBERT embedding backend (`all-MiniLM-L6-v2`, 384 dimensions)  
**Inference Profile**: `fast` (`HuggingFaceTB/SmolLM2-360M-Instruct`, 360M parameters)  
**Execution Device**: `cpu` (Intel Core i5-13450HX)  
**Date Evaluated**: September 5, 2026  

---

## Ingestion Command & Output

```bash
$ aether ingest "e:\LLM\scratch\attention.pdf"
  status: indexed
  docs: 1
  chunks: 91
```

---

## Query 1: Mechanism Description

**Prompt**:
```bash
$ aether ask "What is the Multi-Head Attention mechanism described in this paper?" --rag -p fast --json
```

**Output**:
```json
{
  "domain": "physics",
  "analysis_result": {
    "confidence": 0.6273,
    "generated_text": "According to the document, Multi-Head Attention is a type of attention mechanism used in neural networks for tasks such as language modeling, semantic segmentation, and image captioning. It enables multiple representations of input data to be combined into a unified representation through a hierarchical process. In simpler terms, it's like having multiple eyes on your face instead of just one eye - you get a better view overall because each eye sees something unique about yourself. The idea is to distribute the computation across multiple units so that they can all contribute equally to the final result. This helps reduce the amount of computation required while maintaining high accuracy. So basically, Multi-Head Attention acts as a kind of collective intelligence within the network, allowing it to make decisions based on the contributions of many individual components working together towards a common goal.",
    "sources": [
      {
        "doc_id": "5841f8715e20",
        "score": 0.6520,
        "title": "Provided proper attribution is provided, Google hereby grants permission to",
        "snippet": "This makes it more difficult to learn dependencies between distant positions [12]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolu"
      },
      {
        "doc_id": "5841f8715e20",
        "score": 0.6381,
        "title": "Provided proper attribution is provided, Google hereby grants permission to",
        "snippet": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this. MultiHead("
      },
      {
        "doc_id": "5841f8715e20",
        "score": 0.6173,
        "title": "Provided proper attribution is provided, Google hereby grants permission to",
        "snippet": "The output is computed as a weighted sum 3 Scaled Dot-Product Attention Multi-Head Attention Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention l"
      },
      {
        "doc_id": "5841f8715e20",
        "score": 0.6017,
        "title": "Provided proper attribution is provided, Google hereby grants permission to",
        "snippet": "For each of these we use dk = dv = dmodel/h = 64. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality. 3.2.3 Ap"
      }
    ],
    "citations_valid": true,
    "profile": "fast",
    "device": "cpu"
  },
  "config_used": "HuggingFaceTB/SmolLM2-360M-Instruct",
  "profile": "fast",
  "device": "cpu"
}
```

### Analysis
- **Semantic Retrieval**: Top 4 chunks all directly target the Multi-Head Attention architecture (Sections 3.2.1, 3.2.2, Figure 2).
- **Explanation**: The model correctly explains that Multi-Head Attention allows attending to multiple representations from different subspaces in parallel.

---

## Query 2: Specific Fact / BLEU Score

**Prompt**:
```bash
$ aether ask "What BLEU score did the model achieve on the WMT 2014 English-to-German task?" --rag -p fast --json
```

**Output**:
```json
{
  "domain": "physics",
  "analysis_result": {
    "confidence": 0.6507,
    "generated_text": "The model achieved a BLEU score of 41.0 on the WMT 2014 English-to-German test.",
    "sources": [
      {
        "doc_id": "5841f8715e20",
        "score": 0.6777,
        "title": "Provided proper attribution is provided, Google hereby grants permission to",
        "snippet": "This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score. 6 Results 6.1 Machine Translation On the WMT 2014 English-to-German translation task, the big transf"
      },
      {
        "doc_id": "5841f8715e20",
        "score": 0.6488,
        "title": "Provided proper attribution is provided, Google hereby grants permission to",
        "snippet": "On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the trai"
      },
      {
        "doc_id": "5841f8715e20",
        "score": 0.6441,
        "title": "Provided proper attribution is provided, Google hereby grants permission to",
        "snippet": "Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models. On the WMT 2014"
      },
      {
        "doc_id": "5841f8715e20",
        "score": 0.6323,
        "title": "Provided proper attribution is provided, Google hereby grants permission to",
        "snippet": "Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost."
      }
    ],
    "citations_valid": true,
    "profile": "fast",
    "device": "cpu"
  },
  "config_used": "HuggingFaceTB/SmolLM2-360M-Instruct",
  "profile": "fast",
  "device": "cpu"
}
```

### Analysis
- **Retrieval Performance**: Highest similarity score (0.6777) retrieved the exact section describing the WMT 2014 translation benchmarks.
- **Model Conflation**: The ground truth in the paper is 28.4 BLEU for English-to-German and 41.8 BLEU for English-to-French. The 360M model extracted 41.0 by conflating the adjacent French translation metric with the German question.

---

## Query 3: Out-of-Scope / Unanswerable Query

**Prompt**:
```bash
$ aether ask "What is CRISPR-Cas9 genome editing mechanism?" --rag -p fast --json
```

**Output**:
```json
{
  "domain": "physics",
  "analysis_result": {
    "confidence": 0.2216,
    "generated_text": "CRISPR-Cas9 (Clustered Regularly Interspaced Short Palindromic Repeats/C-terminal Endonuclease 9) is a gene editing technology used in molecular biology and biotechnology. It was discovered by Jennifer Doudna and Emmanuelle Charpentier at the University of California, Berkeley, in 2012. Here's how it works: 1. A small RNA molecule called guide RNA (gRNA) is programmed with specific sequences of nucleotides known as \"target sites.\" These target sites match up with DNA strands in the cell. 2. When cells receive the gRNA, they recognize these target sites and use them as guides to locate and cut the DNA strands near those sites. 3. The gRNA cuts the DNA strands, creating gaps about 20 base pairs long. Then, an enzyme called Cas9 (CRISPR associated protein 9) recognizes the gap and binds to it. 4. The Cas9 enzyme then cuts the DNA strand where the gRNA targeted it, making a double-stranded break. This disruption causes genetic mutations, which may be repaired naturally or corrected by other cellular processes. 5. To edit genes, researchers add new DNA segments into the affected",
    "sources": [
      {
        "doc_id": "5841f8715e20",
        "score": 0.2611,
        "title": "Provided proper attribution is provided, Google hereby grants permission to",
        "snippet": "This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2"
      },
      {
        "doc_id": "5841f8715e20",
        "score": 0.2251,
        "title": "Provided proper attribution is provided, Google hereby grants permission to",
        "snippet": "To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-ali"
      },
      {
        "doc_id": "5841f8715e20",
        "score": 0.2232,
        "title": "Provided proper attribution is provided, Google hereby grants permission to",
        "snippet": "Gomez* University of Toronto aidan@cs.toronto.edu Lukasz Kaiser* Google Brain lukaszkaiser@google.com Illia Polosukhin* illia.polosukhin@gmail.com Abstract The dominant sequence transduction models"
      },
      {
        "doc_id": "5841f8715e20",
        "score": 0.1768,
        "title": "Provided proper attribution is provided, Google hereby grants permission to",
        "snippet": "2 Figure 1: The Transformer - model architecture. The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder"
      }
    ],
    "citations_valid": true,
    "profile": "fast",
    "device": "cpu"
  },
  "config_used": "HuggingFaceTB/SmolLM2-360M-Instruct",
  "profile": "fast",
  "device": "cpu"
}
```

### Analysis
- **Retrieval Confidence Separation**: Top chunk similarity dropped from 0.65+ down to 0.26, and confidence fell to 0.2216.
- **Fall-Through Behavior**: Since the index contained zero CRISPR data, the generator answered exclusively from pre-trained parametric weights.

---

## Summary Metrics

| Query Type | Topic | Top Cosine Similarity | Confidence Score | Hallucination Guard | Groundedness |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mechanism** | Multi-Head Attention | **0.6520** | **0.6273** | Valid (0 stripped) | High Semantic Match |
| **Factual Detail** | WMT 2014 BLEU score | **0.6777** | **0.6507** | Valid (0 stripped) | Contextual Conflation |
| **Out-of-Scope** | CRISPR-Cas9 mechanism | **0.2611** | **0.2216** | Valid (0 stripped) | Parametric Fallback |
