# System Architecture: Hybrid GA-GWO Augmentation Pipeline

This document provides a detailed visual and structural breakdown of the end-to-end incremental natural language processing framework.

## 1. Complete Runtime Pipeline

The entire system is orchestrated to handle imbalanced, low-resource cyber banking text streams by coupling dynamic semantic augmentation with a pre-trained RoBERTa backbone.

```mermaid
graph TD
    subgraph Phase 1: Core Framework Pipeline
        A["Raw Data (Imbalanced)"] --> B{"FACI Calculator<br>(Lexical, Semantic, Entity)"}
        B -->|Computes Complexity| C["Chunking Mechanism"]
        C --> D["Policy Memory<br>(Warm-starts Elite Wolves)"]
        D --> E["Hybrid GA-GWO Optimizer"]
        E -->|Output: Budget & Strategy| F["Augmentor<br>(EDA, Synonym, BERT, Translation)"]
        
        F -->|Generated Candidates| G{"Semantic Validator<br>(Cosine Similarity Filter)"}
        G -->|Rejected| H["Discarded"]
        G -->|Accepted| I["Class-Balanced Replay Buffer<br>(Combats Catastrophic Forgetting)"]
        
        I -->|Balanced Mini-batches| J["RoBERTa Backbone<br>(Incremental Fine-Tuning)"]
        J --> K["Class Balanced Loss (CBL)"]
        K --> L["Model Checkpoint"]
    end
    
    subgraph Phase 2: Evaluation & Feedback
        L --> M{"Test Set Evaluation"}
        M --> N["Final Metrics<br>(Macro F1, Accuracy)"]
        M --> O["Performance Feedback<br>(Updates Policy Memory)"]
        O -.-> D
    end
    
    subgraph Phase 3: Ablation Study & Robustness
        N --> P["Multi-Seed Evaluation Complete"]
        P --> Q{"Systematic Component Ablation"}
        Q -->|Disable Replay Buffer| R1["Evaluate Forgetting"]
        Q -->|Disable FACI / GA / GWO| R2["Evaluate Marginal Impact"]
        R1 --> S["Final Aggregation CSVs"]
        R2 --> S
    end
```

---

## 2. Hybrid Optimization Sequence (GA + GWO)

The core novelty of the framework is the union of the **Genetic Algorithm (GA)** for broad exploration and the **Grey Wolf Optimizer (GWO)** for highly targeted local exploitation of the augmentation policy space.

```mermaid
sequenceDiagram
    participant Orchestrator
    participant PolicyMemory
    participant GeneticAlgorithm
    participant GreyWolfOptimization
    
    Orchestrator->>PolicyMemory: Fetch Historical Elites (Alpha, Beta, Delta)
    PolicyMemory-->>GeneticAlgorithm: Seed Initial Population
    GeneticAlgorithm->>GeneticAlgorithm: Crossover & Mutate (Broad Search)
    GeneticAlgorithm-->>GreyWolfOptimization: Refined Elite Pool (8-dim)
    GreyWolfOptimization->>GreyWolfOptimization: Encircling & Hunting (Alpha, Beta, Delta)
    GreyWolfOptimization-->>Orchestrator: Alpha Wolf (Best Policy + Budget)
    Orchestrator->>PolicyMemory: Save Alpha Policy & Metrics for Next Chunk
```

---

## 3. Core Component Descriptions

### **Feature-Aware Complexity Index (FACI)**
A scalar calculation mapping the intrinsic difficulty of an incoming text chunk based on three pillars: Lexical Diversity, Semantic Ambiguity, and Named Entity density. Higher complexity chunks receive tighter augmentation budgets to preserve structural integrity.

### **Policy Memory**
Maintains a historical ledger of top-performing augmentation policies ("Alpha wolves"). When processing a new chunk, it warm-starts the Genetic Algorithm by seeding it with proven policies to prevent the optimizer from converging slowly from scratch.

### **Augmentor & Semantic Validator**
Executes the selected strategy (EDA, Contextual BERT, Back Translation, or Synonym Replacement) based on the assigned budget. It then filters every candidate augmentation using a cosine similarity threshold against a frozen SentenceTransformer embedding to guarantee the underlying financial intent has not been corrupted.

### **Class-Balanced Replay Buffer**
The most structurally critical component for dealing with the `>3:1` class imbalance in the cyber banking dataset. It retains high-quality real and augmented minority class samples across training chunks to prevent the model from suffering catastrophic forgetting during incremental updates.

### **RoBERTa + Class-Balanced Loss (CBL)**
The primary classification engine. RoBERTa is incrementally fine-tuned using Focal Loss and Class-Balanced inverse frequency weighting to heavily penalize misclassifications on minority complaints.
