# DIDI RL SOAR: The Visual Master Plan 🛡️🧠📊

> [!IMPORTANT]
> **This is the Single Source of Truth.**
> It documents the entire project: The Philosophy, The Architecture, The Training, and The Future.
> **Every section includes a Diagram.**

---

## 1. The Philosophy: "The Infinite Game" ♾️

We moved from **Static Scripts** to **Dynamic Interaction**.

### Concept Diagram
```mermaid
graph LR
    subgraph "Old Way (Static)"
        Alert["Alert: Brute Force"] -->|If/Else| Block["Script: Block IP"]
        NewAttack["Alert: Zero Day"] -->|No Rule| Failure["❌ Breach"]
    end

    subgraph "New Way (Dynamic)"
        Sim["Simulator (Attacks)"] <-->|Interaction| Agent["AI Agent (Defense)"]
        Interaction["Unique Timeline"] -->|Teaches| Observer["Observer (Eyes)"]
        Observer -->|Upgrades| Agent
        Agent -->|Adapts to| AnyAttack["✅ Any Attack"]
    end
```

---

## 2. The Architecture (What We Built) 🏗️

The system has three main components working in a loop.

### System Diagram
```mermaid
graph TD
    subgraph "World (The Environment)"
        Sim["Simulator Logic"] -->|Infects| Net["Network Devices"]
        Net -->|Emits| Logs["Raw Logs & Alerts"]
    end

    subgraph "Stage 1: The Eyes (Observer)"
        Logs -->|Autoencoder| Latent["12D Confidence State"]
        Latent -->|Anomaly Score| Risk["Risk Assessment"]
    end

    subgraph "Stage 2: The Brain (PPO Agent)"
        Latent -->|Policy Network| Decision["Action Selection"]
        Decision -->|Executor| Action["Mitigation Action"]
    end

    Action -->|Fixes| Net
    Action -->|Stops| Sim
```

---

## 3. Component Deep Dive 🔬

### A. The Observer (Vision System)
How it turns "Noise" into "Signal" using a Hybrid Autoencoder.

```mermaid
graph TD
    subgraph "Inputs"
        T["Syslog Text"] & A["Snort Alerts"] & M["Metrics (CPU)"]
    end

    subgraph "Encoder (Compression)"
        T & A & M -->|Neural Nets| Features["Feature Vectors"]
        Features -->|Concat| Joint["Joint Representation"]
        Joint -->|Bottleneck| State["12D Latent State"]
    end

    subgraph "Outputs (Tasks)"
        State -->|Reconstruction| Recon["Input Reconstruction"]
        State -->|Classification| Class["Risk Variance"]
    end

    Recon -->|Error = Anomaly| Anomaly["Check: Is this new?"]
```

### B. The Agent (Decision System)
How it learns Strategy using Actor-Critic PPO.

```mermaid
graph TD
    Input["12D State"] -->|Shared Layers| Hidden["Understanding of Situation"]

    subgraph "The Actor (Doer)"
        Hidden -->|Policy Head| Logits["Action Probabilities"]
        Logits -->|Sample| Action["Selected: Isolate Server"]
    end

    subgraph "The Critic (Judge)"
        Hidden -->|Value Head| Value["Predicted Reward: +10"]
    end

    Reward["Actual Reward"] -->|Compare| Value
    Result["Advantage"] -->|Update| Hidden
```

---

## 4. The Training Pipeline (Status: Running) 🏃‍♂️

We use a **Transfer Learning** approach to build the brain.

### The Learning Flow
```mermaid
graph LR
    Step1["Phase 1: Bootstrapping"] -->|Static Data| Obs["Train Observer"]
    Obs -->|Lock Model| Step2["Phase 2: Specialization"]
    Step2 -->|Simulated interaction| Brain["Train PPO Agent"]
    Brain -->|Active Learning| Step3["Phase 3: Fine-Tuning"]
    Step3 -->|Loop 50x| Mastery["Mastery (Brain + Eyes)"]

    style Step2 fill:#bfb,stroke:#333,stroke-width:2px
    style Step3 fill:#fbb,stroke:#333,stroke-width:2px
```

*   **Phase 1 (Done)**: Initial knowledge.
*   **Phase 2 (Current)**: "Pro Mode" Training (5M Steps).
*   **Phase 3 (Next)**: Intensive Iterative Refinement.

---

## 5. Deployment Strategy (The Active Flywheel) 🎡

How the system gets smarter *after* deployment.

### The Flywheel Diagram
```mermaid
graph TD
    Deploy["Agent v1.0 Live"] -->|High Confidence (>90%)| Auto["Auto-Mitigate"]
    Deploy -->|Low Confidence (<90%)| Review["Human Review Queue"]
    
    Review -->|Human Label| Data["New Training Data"]
    Data -->|Sunday Retrain| Tune["Fine-Tune Model"]
    Tune -->|Sunday Deploy| Upgrade["Agent v2.0 Live"]
    
    Upgrade -->|Better Handling| Deploy
```

---

## 6. Future Roadmap (What Comes Next) 🔮

The concrete steps to clear "Level 5+".

### The Mastery Timeline
```mermaid
gantt
    title The Road to World-Class Security
    dateFormat  X
    axisFormat %d

    section Phase 1: Tuning
    Intensive Fine-Tuning (50 Iterations)   :active, a1, 0, 1d

    section Phase 2: New Threats
    Add Ransomware Scenario                 :a2, after a1, 4h
    Add Stealth Mode Scenario               :a3, after a1, 4h
    Train on New Threats                    :a4, after a2, 12h

    section Phase 3: Perfection
    Reward Shaping (Surgical Defense)       :a5, after a4, 2h
    Deploy Active Flywheel                  :a6, after a5, 4h
```

---

## 7. Operational Commands 💻

### A. Check Status (Current Run)
```bash
tail -f training_full_speed.log
```

### B. Start Phase 1 (Fine-Tuning)
*Whan current run finishes:*
```bash
python3 train/train_iterative.py --iterations 50
```

### C. Evaluate (Visualize)
```bash
python3 eval/animate_eval.py --scenarios all
```
