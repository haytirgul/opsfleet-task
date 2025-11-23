# AI Technical Assignment - LangGraph Helper Agent

## The Challenge

Build an AI LangGraph Helper Agent in Python that helps developers work with LangGraph and LangChain by answering practical questions. You can implement the agent itself using any platform of your choice (preferably LangGraph, but it's not a requirement).

It should support both offline and online modes to handle different usage scenarios, such as working without internet access or integrating with live documentation sources.

## Core Requirements

### 1. Dual Operating Modes

Your agent must support two distinct modes (controlled via environment variable or command-line flag):

#### Offline Mode (Required)

* Works without internet connectivity - Please notice that you are still allowed to use LLMs via API (e.g., Gemini) but not external web-related resources.
* Uses documentation from (download them to use locally) LangGraph llms.txt resources
    * For LangChain V1 you can use: https://docs.langchain.com/llms.txt, https://docs.langchain.com/llms-full.txt
* Must document your data preparation strategy
* If you extend with additional data sources: You must specify how you plan to keep that data up-to-date

#### Online Mode (Required)

* Allows internet connectivity for real-time information
* You may use web searches, APIs, or other online resources of your choice
* All external services must have a free tier (like SERP, EXA, Tavily, DuckDuckGo, etc.) with clear instructions for obtaining API keys in the README file

### 2. Language Model

* Use Google Gemini (free tier)
* Obtain API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
* See [Rate Limits](https://ai.google.dev/pricing)

### 3. Technical Stack

* **Implementation Language**: Python
* **Orchestration Framework**: Platform of your choice (preferably LangGraph, but it's not a requirement) (Python)
* Agent design and architecture are your choice

### 4. Documentation Sources

Start with the llms.txt format documentation:

* **LangGraph Python**: 
    * https://langchain-ai.github.io/langgraph/llms.txt
    * https://langchain-ai.github.io/langgraph/llms-full.txt
* **LangChain Python**: https://python.langchain.com/llms.txt

You may extend this with additional sources, but document your approach.

### 5. Portability

Your solution must be runnable on another machine. This is a key evaluation criterion. Ensure:

* Clear setup instructions
* Dependency management
* Environment configuration guidance

## Deliverables (via GitHub)

Submit a public GitHub repository containing:

### 1. Working Code

* Python implementation
* Support for both offline and online modes
* Mode switching via environment variable or CLI flag

### 2. Documentation (README)

* **Architecture Overview**: Graph design, state management, node structure
* **Operating Modes**:
    * How offline mode works and what data it uses
    * How online mode works and what services it leverages
    * How to switch between modes
* **Data Freshness Strategy**:
    * Offline: How you prepared the data; if extended, how users update it
    * Online: What services you use and why
* **Setup Instructions** and example run

Clear version specifications where important

## Example Questions Your Agent Should Handle

* "How do I add persistence to a LangGraph agent?"
* "What's the difference between StateGraph and MessageGraph?"
* "Show me how to implement human-in-the-loop with LangGraph"
* "How do I handle errors and retries in LangGraph nodes?"
* "What are best practices for state management in LangGraph?"

## Notes

### Free Tier Requirements

Any external service or tool you use must:

* Have a free tier or be completely free
* Include in your README: Service name, why you chose it, and how to obtain API keys (with links, like we did with Gemini)

### Data Update Strategy

If you add data sources beyond the provided llms.txt files:

* Document what you added
* Specify how users can refresh/update this data
* Consider automation or provide clear manual steps

### Mode Switching

Clearly document how users control the mode:

```bash
python main.py --mode offline "How do I use checkpointers?"
```

```bash
export AGENT_MODE=online
python main.py "What are the latest LangGraph features?"
```

## Getting Started Resources

* [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
* [LangGraph llms.txt Overview](https://langchain-ai.github.io/langgraph/llms-txt-overview/)
* [Google AI Studio (Gemini API Keys)](https://aistudio.google.com/app/apikey)
* [LangChain Python Docs](https://python.langchain.com/)
