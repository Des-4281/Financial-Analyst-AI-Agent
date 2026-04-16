# =============================================================================
# AUTONOMOUS FINANCIAL RESEARCH ANALYST — CLI Version
# A LangGraph-powered agent that autonomously gathers stock data, news,
# sentiment, and private analyst reports to generate investment recommendations.
#
# SETUP:
#   1. pip install -r requirements.txt
#   2. python agent.py  
#
# API Keys needed:
#   OpenAI, Tavily (Free Tier)
# =============================================================================

import os
import json
import logging
import zipfile
from typing import Dict, List, Literal, Annotated, Sequence
from datetime import datetime
from typing import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================
def load_config(config_path: str = "config.json") -> dict:
    """
    Load API keys from config.json.
    On first run, prompts the user and saves keys locally.
    config.json is listed in .gitignore and never uploaded to GitHub.
    """
    if not os.path.exists(config_path):
        print("\n" + "="*60)
        print("  FIRST TIME SETUP")
        print("="*60)
        print("You need two API keys to run this agent.")
        print("This is a one-time setup — keys are saved locally")
        print("to config.json and never uploaded to GitHub.\n")
        print("  OpenAI key:  https://platform.openai.com")
        print("  Tavily key:  https://app.tavily.com (free tier)\n")

        config = {
            "API_KEY": input("Enter your OpenAI API key: ").strip(),
            "OPENAI_API_BASE": "https://api.openai.com/v1",
            "TAVILY_API_KEY": input("Enter your Tavily API key: ").strip()
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print("\n✅ Keys saved — you won't be asked again.\n")

    with open(config_path, "r") as f:
        config = json.load(f)

    os.environ["OPENAI_API_KEY"] = config.get("API_KEY", "")
    os.environ["OPENAI_API_BASE"] = config.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    os.environ["TAVILY_API_KEY"] = config.get("TAVILY_API_KEY", "")
    logger.info("✅ Configuration loaded")
    return config


# =============================================================================
# AGENT CHARTER
# =============================================================================
AGENT_CHARTER = """You are an autonomous Financial Research Analyst Agent.

PRIMARY MISSION:
Analyze public companies and generate comprehensive investment research reports.

AVAILABLE TOOLS:
• get_stock_price(ticker) — Current price, volume, market cap
• get_stock_history(ticker, period) — Historical data (use '3y' for 3-year analysis)
• search_financial_news(query) — Real-time news search
• analyze_sentiment(text) — Sentiment score 0.0 to 1.0
• query_private_database(query) — Internal analyst reports (if available)

RULES:
✓ Always gather data proactively — don't wait to be asked
✓ Always check 3-year historical performance
✓ Always search and analyze recent news sentiment
✓ Always query private database if available
✓ Always cite every claim with source and timestamp
✓ Always give a clear Buy/Hold/Sell with confidence %
✓ Always identify at least 3 risk factors
✓ If a tool fails, use alternatives and note the gap — never stop

REPORT FORMAT:
1. Executive Summary (2-3 sentences)
2. Financial Metrics (price, market cap, 3-year return %)
3. Sentiment Analysis (score, headlines with URLs)
4. AI Research Activity (from private database if available)
5. Risk Factors (minimum 3, include geopolitical where relevant)
6. Investment Recommendation (Buy/Hold/Sell, confidence %, rationale)
7. Source Citations
8. Gaps & Limitations
"""


# =============================================================================
# TOOLS
# =============================================================================
@tool
def get_stock_price(ticker: str) -> Dict:
    """
    Returns current stock price and key metrics for a given ticker.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'NVDA')

    Returns:
        dict with current_price, market_cap, volume, day range, timestamp
    """
    try:
        info = yf.Ticker(ticker.upper()).info
        price = (info.get('currentPrice') or
                 info.get('regularMarketPrice') or
                 info.get('previousClose'))
        if not price:
            return {'ticker': ticker.upper(), 'status': 'error',
                    'error': f'No price data for {ticker}'}
        return {
            'ticker': ticker.upper(),
            'company_name': info.get('longName', ticker),
            'current_price': round(price, 2),
            'currency': info.get('currency', 'USD'),
            'day_high': info.get('dayHigh'),
            'day_low': info.get('dayLow'),
            'volume': info.get('volume'),
            'market_cap': info.get('marketCap'),
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
    except Exception as e:
        return {'ticker': ticker.upper(), 'status': 'error', 'error': str(e),
                'timestamp': datetime.now().isoformat()}


@tool
def get_stock_history(ticker: str, period: str = "1y") -> Dict:
    """
    Returns historical stock performance over a given period.

    Args:
        ticker: Stock ticker symbol
        period: '1mo', '3mo', '6mo', '1y', '2y', '3y', '5y' — default '1y'

    Returns:
        dict with return_pct, start/end prices, high, low, avg_volume
    """
    try:
        hist = yf.Ticker(ticker.upper()).history(period=period)
        if hist.empty:
            return {'ticker': ticker.upper(), 'status': 'error',
                    'error': f'No data for {ticker} over {period}'}
        start = hist['Close'].iloc[0]
        end = hist['Close'].iloc[-1]
        return {
            'ticker': ticker.upper(),
            'period': period,
            'start_date': hist.index[0].strftime('%Y-%m-%d'),
            'end_date': hist.index[-1].strftime('%Y-%m-%d'),
            'start_price': round(start, 2),
            'end_price': round(end, 2),
            'return_pct': round(((end - start) / start) * 100, 2),
            'high': round(hist['High'].max(), 2),
            'low': round(hist['Low'].min(), 2),
            'avg_volume': int(hist['Volume'].mean()),
            'status': 'success'
        }
    except Exception as e:
        return {'ticker': ticker.upper(), 'status': 'error', 'error': str(e)}


@tool
def search_financial_news(query: str) -> List[Dict]:
    """
    Searches real-time financial news.

    Args:
        query: Search string e.g. 'Apple AI strategy 2025'

    Returns:
        List of articles with title, url, content, score
    """
    try:
        tavily = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
            include_images=False
        )
        return tavily.invoke(query)
    except Exception as e:
        return [{'status': 'error', 'error': str(e)}]


@tool
def analyze_sentiment(text: str) -> Dict:
    """
    Analyzes financial text sentiment using GPT-4o-mini.

    Args:
        text: Headline, article excerpt, or any financial text

    Returns:
        dict with sentiment, score (0-1), confidence, reasoning
    """
    try:
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = f"""Analyze the sentiment of this financial text and respond ONLY with valid JSON:
{{
    "sentiment": "positive|negative|neutral",
    "score": 0.0-1.0,
    "confidence": 0.0-1.0,
    "reasoning": "one sentence"
}}

Text: {text}"""
        result = json.loads(model.invoke(prompt).content)
        result['status'] = 'success'
        return result
    except Exception as e:
        pos_words = ['growth', 'profit', 'gain', 'success', 'strong', 'beat', 'record']
        neg_words = ['loss', 'decline', 'down', 'weak', 'risk', 'concern', 'miss']
        tl = text.lower()
        pos = sum(1 for w in pos_words if w in tl)
        neg = sum(1 for w in neg_words if w in tl)
        if pos > neg:
            s, sc = 'positive', min(1.0, 0.6 + pos * 0.05)
        elif neg > pos:
            s, sc = 'negative', max(0.0, 0.4 - neg * 0.05)
        else:
            s, sc = 'neutral', 0.5
        return {'sentiment': s, 'score': sc, 'confidence': 0.5,
                'reasoning': 'Keyword fallback', 'status': 'fallback', 'note': str(e)}


# =============================================================================
# RAG (optional private knowledge base)
# =============================================================================
_retriever = None


def setup_rag(pdf_dir: str = "Companies-AI-Initiatives",
              zip_path: str = "Companies-AI-Initiatives.zip") -> bool:
    """
    Indexes PDF analyst reports for private knowledge base queries.
    Completely optional — agent works perfectly without it.
    """
    global _retriever

    if not os.path.exists(pdf_dir) and os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(".")
        logger.info(f"✅ Extracted {zip_path}")

    if not os.path.exists(pdf_dir) or not any(f.endswith('.pdf') for f in os.listdir(pdf_dir)):
        logger.info("ℹ️  No PDFs found — private knowledge base disabled")
        return False

    pdfs = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    logger.info(f"📄 Loading {len(pdfs)} analyst report(s)...")

    chunks = PyPDFDirectoryLoader(path=f"{pdf_dir}/").load_and_split(
        text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=1000, chunk_overlap=200
        )
    )
    _retriever = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
        collection_name="analyst_reports"
    ).as_retriever(search_type="similarity", search_kwargs={"k": 10})

    logger.info(f"✅ Private knowledge base ready ({len(chunks)} chunks)")
    return True


@tool
def query_private_database(query: str) -> str:
    """
    Query private analyst reports for company AI initiative details.

    Args:
        query: Natural language question about a company's AI strategy

    Returns:
        str: Answer from analyst reports, or unavailability message
    """
    if _retriever is None:
        return "Private knowledge base not available — no analyst reports loaded."
    try:
        context = ". ".join([c.page_content for c in _retriever.invoke(query)])
        prompt = f"""Answer using ONLY the context below. Cite which company each fact is from.
If not in context, say "Not available in analyst reports."

Context: {context}

Question: {query}"""
        return ChatOpenAI(model="gpt-4o-mini", temperature=0).invoke(prompt).content
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# AGENT
# =============================================================================
class AgentState(TypedDict):
    messages: Annotated[Sequence, add_messages]


def build_agent(use_rag: bool = True, use_memory: bool = True):
    """
    Builds the financial analyst agent.

    Args:
        use_rag: Include private database tool (requires PDF setup)
        use_memory: Remember conversation history across turns

    Returns:
        Compiled LangGraph agent
    """
    tools = [get_stock_price, get_stock_history, search_financial_news, analyze_sentiment]
    if use_rag and _retriever is not None:
        tools.append(query_private_database)

    logger.info(f"📦 Building agent — {len(tools)} tools, "
                f"RAG={'on' if use_rag and _retriever else 'off'}, "
                f"memory={'on' if use_memory else 'off'}")

    model_with_tools = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        messages = [SystemMessage(content=AGENT_CHARTER)] + list(state["messages"])
        response = model_with_tools.invoke(messages)
        if hasattr(response, 'tool_calls') and response.tool_calls:
            logger.info(f"🔧 Calling: {[tc['name'] for tc in response.tool_calls]}")
        else:
            logger.info("✅ Final response ready")
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        return "tools" if (hasattr(last, 'tool_calls') and last.tool_calls) else "end"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")

    graph = workflow.compile(checkpointer=MemorySaver()) if use_memory else workflow.compile()
    logger.info("✅ Agent ready\n")
    return graph


def ask(agent, query: str, thread_id: str = "default") -> str:
    """Send a query to the agent and return the response."""
    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": thread_id}}
    )
    return result["messages"][-1].content


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    load_config()
    rag_available = setup_rag()
    agent = build_agent(use_rag=rag_available, use_memory=True)

    print("\n" + "="*70)
    print("  AUTONOMOUS FINANCIAL RESEARCH ANALYST")
    print("="*70)
    print("Type any query below. Type 'quit' to exit.\n")
    print("Examples:")
    print("  • Analyze Apple stock (AAPL)")
    print("  • Compare NVIDIA and AMD for AI investment")
    print("  • Rank MSFT, GOOGL, NVDA, AMZN by investment potential")
    print("  • What are the risks in Tesla right now?")
    if rag_available:
        print("  • What AI projects is Microsoft working on?")
    print("="*70 + "\n")

    while True:
        try:
            query = input("You: ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            print("\n🔍 Analyzing... (30-60 seconds)\n")
            print("🤖 Agent:\n")
            print(ask(agent, query))
            print("\n" + "-"*70 + "\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

