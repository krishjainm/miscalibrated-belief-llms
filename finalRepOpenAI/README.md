# WMAC 2026 Research: Emergent Communication Protocols in Multi-Agent Poker

## 🎯 Research Overview

This repository contains the complete implementation and analysis framework for our WMAC 2026 research on **Emergent Communication Protocols in Multi-Agent Texas Hold'em Poker**. The research demonstrates how LLM agents can develop sophisticated coordination strategies through emergent communication without predefined protocols.

## 📊 Key Research Contributions

### **1. Emergent Communication Discovery**
- **Protocol Emergence**: LLM agents develop their own communication strategies without hardcoded signals
- **Lexical Adaptation**: Agents adapt language when specific phrases are banned, demonstrating protocol robustness
- **Coordination Effectiveness**: Measurable team advantage through emergent communication

### **2. Research-Neutral Framework**
- **No Hardcoded Strategies**: Prompts guide strategic thinking without fixed thresholds or actions
- **Emergent Discovery**: Agents discover coordination opportunities through natural language interaction
- **Adaptive Communication**: Dynamic protocol development based on game context

### **3. Comprehensive Analysis Pipeline**
- **Protocol Analysis**: Message-action coupling, protocol evolution, signal detection
- **Performance Analysis**: Team advantage, coordination effectiveness, communication efficiency
- **Temporal Analysis**: Communication patterns over time and game phases
- **Live Metrics**: Real-time coordination tracking and adaptation measurement

## 🏗️ Repository Structure

```
finalRepOpenAI/
├── game_environment/          # Core game logic and agent implementations
│   ├── advanced_collusion_agent.py
│   ├── mixed_player_communication_game.py
│   ├── llm_agent.py
│   └── ...
├── texasholdem/               # Texas Hold'em game engine
├── utils/                     # Utility functions and logging
├── wmac2026/                  # WMAC 2026 research framework
│   ├── run_wmac.py           # Main simulation runner
│   ├── prompt_library.py     # Research-compliant prompts
│   └── ...
├── analysis/                  # Analysis pipeline
│   ├── analysis_pipeline.py
│   ├── protocol_analysis.py
│   ├── performance_analysis.py
│   └── ...
├── data/                      # Simulation data and results
└── README.md                  # This file
```

## 🚀 Quick Start

### **1. Setup Environment**
```bash
cd finalRepOpenAI
pip install -r requirements.txt
```

### **2. Run Basic Simulation**
```bash
# 20-hand game with 2 colluding LLM agents
python3 wmac2026/run_wmac.py --hands 20 --collusion-llm-players 2 --coordination-mode emergent_only --output-dir data/simulation_1
```

### **3. Run Robustness Testing**
```bash
# Test protocol adaptation with banned phrases
python3 wmac2026/run_wmac.py --hands 20 --collusion-llm-players 2 --coordination-mode emergent_only --ban-phrases "build,building,support" --enforce-bans --output-dir data/simulation_2
```

### **4. Run Complete Analysis**
```bash
# Analyze all simulation data
python3 analysis/run_complete_analysis.py
```

## 🔬 Research Methodology

### **Phase 1: Baseline Emergent Communication**
- **Objective**: Establish baseline protocol development
- **Configuration**: 2 colluding LLM agents, emergent_only mode
- **Metrics**: Message patterns, coordination effectiveness, team performance

### **Phase 2: Robustness Testing**
- **Objective**: Test protocol adaptation under constraints
- **Configuration**: Banned phrase testing with enforcement
- **Metrics**: Lexical adaptation, protocol robustness, semantic preservation

### **Phase 3: Extended Testing**
- **Objective**: Validate scalability and stability
- **Configuration**: Larger games (50+ hands), multiple players
- **Metrics**: Protocol stability, coordination consistency, performance maintenance

## 📈 Key Findings

### **Protocol Emergence**
- **Message-Action Coupling**: Strong correlation between communication and strategic actions
- **Coordination Sequences**: Systematic development of coordination patterns
- **Signal Detection**: Effective identification of coordination signals vs. noise

### **Lexical Adaptation**
- **Banned Phrase Replacement**: Successful adaptation when specific phrases are banned
- **Semantic Preservation**: Meaning maintained while changing vocabulary
- **Protocol Robustness**: Coordination effectiveness maintained under constraints

### **Team Performance**
- **Chip Advantage**: Measurable team advantage through coordination
- **Coordination Effectiveness**: Strong correlation between communication and strategic outcomes
- **Comparative Performance**: Significant advantage over non-coordinating players

## 🛠️ Advanced Usage

### **Custom Simulation Parameters**
```bash
# Extended testing with larger games
python3 wmac2026/run_wmac.py --hands 50 --collusion-llm-players 2 --coordination-mode emergent_only --output-dir data/extended_test

# Multi-player coordination
python3 wmac2026/run_wmac.py --hands 30 --collusion-llm-players 3 --coordination-mode emergent_only --output-dir data/multiplayer_test
```

### **Analysis Pipeline**
```bash
# Run specific analysis components
python3 analysis/protocol_analysis.py
python3 analysis/performance_analysis.py
python3 analysis/temporal_analysis.py
```

## 📊 Data Structure

### **Simulation Output**
Each simulation generates:
- `simulation_meta.json`: Configuration and final statistics
- `chat_dataset/`: Complete communication logs
- `hand_logs/`: Detailed game state and action logs

### **Analysis Output**
- `protocol_analysis.txt`: Message-action coupling and protocol evolution
- `performance_analysis.txt`: Team advantage and coordination metrics
- `temporal_analysis.txt`: Time-based communication patterns
- `RESEARCH_ANALYSIS_SUMMARY.md`: Comprehensive research findings

## 🔧 Configuration Options

### **Coordination Modes**
- `emergent_only`: Pure emergent communication (research-compliant)
- `explicit`: Direct coordination with predefined strategies
- `advisory`: Coordination suggestions without enforcement

### **Communication Styles**
- `emergent`: Natural language coordination
- `emergent_discovery`: Protocol discovery mode
- `emergent_adaptive`: Adaptive communication based on context

### **Testing Parameters**
- `--hands`: Number of hands per simulation
- `--collusion-llm-players`: Number of coordinating agents
- `--ban-phrases`: Phrases to ban for robustness testing
- `--enforce-bans`: Enable phrase banning enforcement

## 📚 Research Papers

This repository supports the following research publications:

1. **WMAC 2026**: "Emergent Communication Protocols in Multi-Agent Poker"
2. **Protocol Analysis**: Message-action coupling and coordination effectiveness
3. **Robustness Testing**: Lexical adaptation and protocol stability
4. **Performance Evaluation**: Team advantage and strategic coordination

## 🤝 Contributing

This is a research repository for WMAC 2026. For questions or collaboration, please refer to the research methodology documentation.

## 📄 License

Research code for academic purposes. See individual files for specific licensing information.

## 📞 Contact

For research inquiries or technical questions, please refer to the WMAC 2026 research methodology documentation.

---

**Research Status**: Complete and ready for publication
**Last Updated**: October 2024
**WMAC 2026 Submission**: Ready