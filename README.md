# AI-agent-based-Deep-Research
```markdown
# DeepResearchAI: Autonomous Research Agent 🤖🔍

**Dual-agent system for automated web research & report generation**  
*Powered by LangGraph, LangChain, and Tavily*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)

## Features
- **Dual-Agent Architecture**: Research (web crawling) & Drafting (report generation) agents
- **Self-Validation**: Auto-quality checks with 3-stage max iteration
- **State Management**: LangGraph-powered workflow tracking
- **Multi-Source Analysis**: Aggregates 5+ web sources per query

## Quick Start
```bash
pip install langgraph langchain-openai tavily-python
```
```python
from research_system import run_research

report = run_research("Compare AI chip advancements: NVIDIA H100 vs Google TPU v5")
```

## Workflow
1. **Research Agent**:  
   - Web crawls with Tavily API
   - Filters relevant content
2. **Drafting Agent**:  
   - Generates structured reports using GPT-4
   - Formats technical comparisons
3. **Validation**:  
   - Checks relevance/accuracy
   - Triggers re-research if needed

## Configuration
```python
os.environ["TAVILY_API_KEY"] = "your_key"  # Get from tavily.com
os.environ["OPENAI_API_KEY"] = "your_key"  # From platform.openai.com
```

## Architecture
```
Query → Research → Draft → Validate → Report  
          ↑____________↗
```
