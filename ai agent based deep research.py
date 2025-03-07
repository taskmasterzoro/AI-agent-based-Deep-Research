import os
from typing import TypedDict, List, Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Initialize components
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.7)
tavily_tool = TavilySearchResults(max_results=5)

# Define state structure
class ResearchState(TypedDict):
    query: str
    content: List[str]
    answer: str
    status: Literal["valid", "invalid"]
    iterations: int

# Research Agent
def research_agent(state: ResearchState):
    print(f"\n🔍 Research Iteration #{state.get('iterations', 0)+1}")
    results = tavily_tool.invoke({"query": state["query"]})
    documents = [f"Source {i+1}:\n{result['content']}" 
                for i, result in enumerate(results)]
    return {"content": documents, "iterations": state.get("iterations", 0)+1}

# Answer Drafter Agent
def draft_answer_agent(state: ResearchState):
    print("📝 Drafting Comprehensive Answer...")
    prompt = ChatPromptTemplate.from_template(
        "As a senior research analyst, create a detailed response to: {query}\n\n"
        "Research Materials:\n{content}\n\n"
        "Include technical specifications, market trends, and expert opinions. "
        "Structure with clear sections and bullet points where appropriate."
    )
    
    chain = (
        RunnablePassthrough.assign(content=lambda x: "\n\n".join(x["content"]))
        | prompt
        | llm
        | StrOutputParser()
    )
    
    answer = chain.invoke({"query": state["query"], "content": state["content"]})
    return {"answer": answer}

# Validation Agent
def validation_agent(state: ResearchState):
    print("🛂 Validating Answer Quality...")
    prompt = ChatPromptTemplate.from_template(
        "Analyze this answer against the original query:\n\n"
        "Query: {query}\n\nAnswer: {answer}\n\n"
        "Check for:\n1. Relevance to query\n2. Source consistency\n3. Technical accuracy\n"
        "Return ONLY 'valid' or 'invalid'"
    )
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke(state).lower()
    status = "valid" if "valid" in result else "invalid"
    
    print(f"Validation Status: {status.upper()}")
    return {"status": status}

# Create workflow
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("research", research_agent)
workflow.add_node("draft_answer", draft_answer_agent)
workflow.add_node("validate", validation_agent)

# Set initial workflow
workflow.set_entry_point("research")
workflow.add_edge("research", "draft_answer")
workflow.add_edge("draft_answer", "validate")

# Add conditional edges
def should_continue(state):
    if state["status"] == "valid" or state.get("iterations", 0) >= 3:
        return END
    return "research"

workflow.add_conditional_edges(
    "validate",
    should_continue,
    {"research": "research", END: END}
)

# Compile the graph
research_app = workflow.compile()

# Run the system
def run_research(query: str):
    results = research_app.invoke({
        "query": query,
        "content": [],
        "answer": "",
        "status": "invalid",
        "iterations": 0
    })
    return results["answer"]

# Example execution
if __name__ == "__main__":
    os.environ["TAVILY_API_KEY"] = "your_tavily_key"
    os.environ["OPENAI_API_KEY"] = "your_openai_key"
    
    complex_query = (
        "Compare NVIDIA's H100 GPU with Google's TPU v5 in terms of: "
        "1. Architecture design\n2. Energy efficiency\n3. LLM training performance\n"
        "4. Current market adoption\n5. Price-performance ratio"
    )
    
    final_answer = run_research(complex_query)
    
    print("\n\n🏁 FINAL RESEARCH REPORT:")
    print(final_answer)