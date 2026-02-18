graph TD
    subgraph "External World"
        Logs["Raw logs / Alerts"]
        Telem["Network Telemetry"]
    end

    subgraph "Perception (Sensory Cortex)"
        DeepEyes["TrainableObserver (Mamba/Transformer)"]
    end

    subgraph "Memory (The Vault)"
        VectorDB["Episodic Memory (Vector DB - Short Term)"]
        KnowledgeGraph["Semantic Memory (Graph DB - Long Term)"]
    end

    subgraph "Logic & Alignment (Prefrontal Cortex)"
        SymbolicRules["Symbolic Policy Engine (Hard Rules)"]
    end

    subgraph "The Brain Core (Decision Hub)"
        WorldModel["Imagination Hub (Hidden State Predictor)"]
        PPOAgent["Decision Module (PPO Agent)"]
    end

    %% Flow
    Logs & Telem --> DeepEyes
    DeepEyes -->|Embeddings| VectorDB
    DeepEyes -->|Features| PPOAgent
    
    KnowledgeGraph -->|Context| PPOAgent
    VectorDB -->|Similar Historical Cases| PPOAgent
    
    PPOAgent -->|Proposed Action| WorldModel
    WorldModel -->|Imagined Reward| PPOAgent
    
    PPOAgent -->|Filtered Action| SymbolicRules
    SymbolicRules -->|Validated Command| World_Exec["Execution Output"]
