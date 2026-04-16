# =============================================================================
# AUTONOMOUS FINANCIAL RESEARCH ANALYST — Web UI
# Streamlit app for Hugging Face
#
# HOW TO DEPLOY:
#   1. Connect this GitHub repo to a Hugging Face Space (Streamlit SDK)
#   2. Add API keys to HF Secrets
#   3. HF auto-deploys and hosts the app. 
# =============================================================================

import os
import json
import logging
import zipfile
from typing import Dict, List, Literal, Annotated, Sequence
from datetime import datetime, timedelta
from typing import TypedDict

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf

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

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================
st.set_page_config(
    page_title="Financial Research Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0d1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Headers */
    h1 { color: #e6edf3 !important; font-family: 'Georgia', serif !important; }
    h2, h3 { color: #c9d1d9 !important; }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        margin-bottom: 8px;
    }

    /* Input */
    [data-testid="stChatInput"] input {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border-color: #30363d !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
    }
    [data-testid="stMetricValue"] { color: #58a6ff !important; }
    [data-testid="stMetricLabel"] { color: #8b949e !important; }
    [data-testid="stMetricDelta"] { font-size: 0.85em !important; }

    /* Buttons */
    .stButton button {
        background-color: #21262d !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
    }
    .stButton button:hover {
        background-color: #30363d !important;
        border-color: #58a6ff !important;
    }

    /* Divider */
    hr { border-color: #30363d !important; }

    /* Text */
    p, li { color: #c9d1d9 !important; }
    .stCaption { color: #8b949e !important; }

    /* Success/warning/info */
    .stSuccess { background-color: #0d2a0d !important; }
    .stWarning { background-color: #2a1f0d !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown("# 📊 Financial Research Analyst")
    st.caption("Autonomous AI agent — live stock data · real-time news · sentiment analysis · investment recommendations")
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Source-181717?logo=github)](https://github.com/your-username/financial-analyst-agent)")

st.divider()

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### 🔑 API Keys")
    st.caption("Used only for your session. Never stored or logged.")

    openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    tavily_key = st.text_input("Tavily API Key", type="password", placeholder="tvly-...")

    keys_ok = bool(openai_key and tavily_key)

    if keys_ok:
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
        os.environ["TAVILY_API_KEY"] = tavily_key
        st.success("✅ Ready")
    else:
        st.warning("Enter both keys to start")
        st.markdown("- [Get OpenAI key](https://platform.openai.com)")
        st.markdown("- [Get Tavily key](https://app.tavily.com) *(free)*")

    st.divider()

    st.markdown("### 📈 Quick Analysis")
    quick_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META", "IBM"]
    selected_ticker = st.selectbox("Quick chart", ["Select..."] + quick_tickers)
    chart_period = st.select_slider("Period", ["1mo", "3mo", "6mo", "1y", "2y", "3y"], value="1y")

    st.divider()

    st.markdown("### 💡 Example Queries")
    examples = [
        "Analyze Apple (AAPL)",
        "Compare NVIDIA vs AMD",
        "Rank MSFT, GOOGL, NVDA, AMZN",
        "Tesla risks right now",
        "Best AI stock to buy?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"btn_{ex}"):
            st.session_state.pending_query = ex

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# =============================================================================
# CHART HELPERS
# =============================================================================
def plot_stock_chart(ticker: str, period: str = "1y") -> go.Figure:
    """Generate a candlestick + volume chart for a ticker."""
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            return None

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25]
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'], high=hist['High'],
            low=hist['Low'], close=hist['Close'],
            name=ticker,
            increasing_line_color='#3fb950',
            decreasing_line_color='#f85149',
        ), row=1, col=1)

        # 20-day moving average
        hist['MA20'] = hist['Close'].rolling(20).mean()
        hist['MA50'] = hist['Close'].rolling(50).mean()
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['MA20'],
            name='MA 20', line=dict(color='#58a6ff', width=1.5, dash='dot')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist['MA50'],
            name='MA 50', line=dict(color='#d29922', width=1.5, dash='dash')
        ), row=1, col=1)

        # Volume bars
        colors = ['#3fb950' if c >= o else '#f85149'
                  for c, o in zip(hist['Close'], hist['Open'])]
        fig.add_trace(go.Bar(
            x=hist.index, y=hist['Volume'],
            name='Volume', marker_color=colors, opacity=0.7
        ), row=2, col=1)

        fig.update_layout(
            title=dict(text=f"{ticker} — {period} Price Chart", font=dict(color='#e6edf3', size=16)),
            paper_bgcolor='#0d1117',
            plot_bgcolor='#161b22',
            font=dict(color='#8b949e'),
            xaxis_rangeslider_visible=False,
            legend=dict(bgcolor='#161b22', bordercolor='#30363d', borderwidth=1),
            height=500,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        fig.update_xaxes(gridcolor='#21262d', showgrid=True)
        fig.update_yaxes(gridcolor='#21262d', showgrid=True)

        return fig
    except Exception:
        return None


def plot_comparison_chart(tickers: list, period: str = "1y") -> go.Figure:
    """Normalized price comparison chart for multiple tickers."""
    try:
        fig = go.Figure()
        colors = ['#58a6ff', '#3fb950', '#d29922', '#bc8cff', '#f85149']

        for i, ticker in enumerate(tickers[:5]):
            hist = yf.Ticker(ticker).history(period=period)
            if hist.empty:
                continue
            normalized = (hist['Close'] / hist['Close'].iloc[0]) * 100
            fig.add_trace(go.Scatter(
                x=hist.index, y=normalized,
                name=ticker,
                line=dict(color=colors[i % len(colors)], width=2)
            ))

        fig.update_layout(
            title=dict(text="Normalized Price Comparison (base 100)",
                       font=dict(color='#e6edf3', size=16)),
            paper_bgcolor='#0d1117',
            plot_bgcolor='#161b22',
            font=dict(color='#8b949e'),
            legend=dict(bgcolor='#161b22', bordercolor='#30363d', borderwidth=1),
            yaxis_title="Return (%)",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        fig.update_xaxes(gridcolor='#21262d')
        fig.update_yaxes(gridcolor='#21262d')
        return fig
    except Exception:
        return None


def get_quick_metrics(ticker: str) -> dict:
    """Fetch key metrics for the dashboard header."""
    try:
        info = yf.Ticker(ticker).info
        price = info.get('currentPrice') or info.get('regularMarketPrice')
        prev = info.get('previousClose', price)
        change_pct = ((price - prev) / prev * 100) if prev else 0
        return {
            'name': info.get('longName', ticker),
            'price': price,
            'change_pct': change_pct,
            'market_cap': info.get('marketCap'),
            'volume': info.get('volume'),
            'pe_ratio': info.get('trailingPE'),
            '52w_high': info.get('fiftyTwoWeekHigh'),
            '52w_low': info.get('fiftyTwoWeekLow'),
        }
    except Exception:
        return {}


# =============================================================================
# QUICK CHART PANEL (sidebar selection)
# =============================================================================
if selected_ticker and selected_ticker != "Select..." and keys_ok:
    with st.expander(f"📈 {selected_ticker} Chart — {chart_period}", expanded=True):
        metrics = get_quick_metrics(selected_ticker)
        if metrics:
            m1, m2, m3, m4 = st.columns(4)
            price = metrics.get('price', 0)
            change = metrics.get('change_pct', 0)
            m1.metric("Price", f"${price:,.2f}", f"{change:+.2f}%")
            mc = metrics.get('market_cap', 0)
            m2.metric("Market Cap", f"${mc/1e12:.2f}T" if mc > 1e12 else f"${mc/1e9:.1f}B")
            m3.metric("52W High", f"${metrics.get('52w_high', 0):,.2f}")
            m4.metric("52W Low", f"${metrics.get('52w_low', 0):,.2f}")

        fig = plot_stock_chart(selected_ticker, chart_period)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

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
✓ Always gather data proactively
✓ Always check 3-year historical performance
✓ Always search and analyze recent news sentiment
✓ Always cite every claim with source and timestamp
✓ Always give Buy/Hold/Sell with confidence %
✓ Always identify at least 3 risk factors including geopolitical ones where relevant
✓ If a tool fails, use alternatives and note the gap

REPORT FORMAT:
1. Executive Summary (2-3 sentences)
2. Financial Metrics (price, market cap, 3-year return %)
3. Sentiment Analysis (score, key headlines with URLs)
4. Risk Factors (minimum 3)
5. Investment Recommendation (Buy/Hold/Sell, confidence %, rationale)
6. Source Citations
7. Gaps & Limitations
"""

# =============================================================================
# TOOLS
# =============================================================================
@tool
def get_stock_price(ticker: str) -> Dict:
    """Returns current stock price and key metrics."""
    try:
        info = yf.Ticker(ticker.upper()).info
        price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        if not price:
            return {'ticker': ticker.upper(), 'status': 'error', 'error': f'No price data for {ticker}'}
        return {
            'ticker': ticker.upper(), 'company_name': info.get('longName', ticker),
            'current_price': round(price, 2), 'currency': info.get('currency', 'USD'),
            'day_high': info.get('dayHigh'), 'day_low': info.get('dayLow'),
            'volume': info.get('volume'), 'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'), 'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
    except Exception as e:
        return {'ticker': ticker.upper(), 'status': 'error', 'error': str(e)}


@tool
def get_stock_history(ticker: str, period: str = "1y") -> Dict:
    """Returns historical stock performance."""
    try:
        hist = yf.Ticker(ticker.upper()).history(period=period)
        if hist.empty:
            return {'ticker': ticker.upper(), 'status': 'error', 'error': f'No data for {ticker}'}
        start, end = hist['Close'].iloc[0], hist['Close'].iloc[-1]
        return {
            'ticker': ticker.upper(), 'period': period,
            'start_date': hist.index[0].strftime('%Y-%m-%d'),
            'end_date': hist.index[-1].strftime('%Y-%m-%d'),
            'start_price': round(start, 2), 'end_price': round(end, 2),
            'return_pct': round(((end - start) / start) * 100, 2),
            'high': round(hist['High'].max(), 2), 'low': round(hist['Low'].min(), 2),
            'avg_volume': int(hist['Volume'].mean()), 'status': 'success'
        }
    except Exception as e:
        return {'ticker': ticker.upper(), 'status': 'error', 'error': str(e)}


@tool
def search_financial_news(query: str) -> List[Dict]:
    """Searches real-time financial news."""
    try:
        return TavilySearchResults(max_results=5, search_depth="advanced",
                                   include_answer=True, include_raw_content=False,
                                   include_images=False).invoke(query)
    except Exception as e:
        return [{'status': 'error', 'error': str(e)}]


@tool
def analyze_sentiment(text: str) -> Dict:
    """Analyzes sentiment of financial text."""
    try:
        prompt = f"""Analyze sentiment. Respond ONLY with valid JSON:
{{"sentiment":"positive|negative|neutral","score":0.0-1.0,"confidence":0.0-1.0,"reasoning":"one sentence"}}
Text: {text}"""
        result = json.loads(ChatOpenAI(model="gpt-4o-mini", temperature=0).invoke(prompt).content)
        result['status'] = 'success'
        return result
    except Exception as e:
        return {'sentiment': 'neutral', 'score': 0.5, 'confidence': 0.3,
                'reasoning': 'Analysis unavailable', 'status': 'error', 'note': str(e)}


@tool
def query_private_database(query: str) -> str:
    """Query private analyst reports."""
    retriever = st.session_state.get('retriever')
    if not retriever:
        return "Private knowledge base not available."
    try:
        context = ". ".join([c.page_content for c in retriever.invoke(query)])
        prompt = f"Answer using ONLY this context. Cite company names.\nContext: {context}\nQuestion: {query}"
        return ChatOpenAI(model="gpt-4o-mini", temperature=0).invoke(prompt).content
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# AGENT
# =============================================================================
class AgentState(TypedDict):
    messages: Annotated[Sequence, add_messages]


def build_agent():
    tools = [get_stock_price, get_stock_history, search_financial_news,
             analyze_sentiment, query_private_database]
    model_with_tools = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        messages = [SystemMessage(content=AGENT_CHARTER)] + list(state["messages"])
        return {"messages": [model_with_tools.invoke(messages)]}

    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        return "tools" if (hasattr(last, 'tool_calls') and last.tool_calls) else "end"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")
    return workflow.compile(checkpointer=MemorySaver())


# =============================================================================
# SESSION STATE
# =============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "session_1"
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "agent" not in st.session_state:
    st.session_state.agent = None

# Build agent once per session when keys are provided
if keys_ok and st.session_state.agent is None:
    with st.spinner("Initializing agent..."):
        st.session_state.agent = build_agent()

# =============================================================================
# CHAT HISTORY
# =============================================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Render charts stored with messages
        if "charts" in msg:
            for chart_data in msg["charts"]:
                if chart_data["type"] == "single":
                    fig = plot_stock_chart(chart_data["ticker"], chart_data["period"])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                elif chart_data["type"] == "comparison":
                    fig = plot_comparison_chart(chart_data["tickers"], chart_data["period"])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# CHART AUTO-DETECTION
# =============================================================================
def extract_tickers(text: str) -> list:
    """
    Roughly extract ticker symbols mentioned in a query.
    Looks for uppercase words 2-5 chars that look like tickers.
    """
    import re
    known = {"AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META",
             "IBM", "AMD", "INTC", "ORCL", "CRM", "NFLX", "BABA", "UBER"}
    found = re.findall(r'\b([A-Z]{2,5})\b', text)
    return [t for t in found if t in known]


def should_show_chart(query: str) -> bool:
    chart_keywords = ["analyze", "chart", "price", "stock", "performance",
                      "history", "trend", "compare", "rank", "investment"]
    return any(k in query.lower() for k in chart_keywords)


# =============================================================================
# INPUT
# =============================================================================
prompt = st.session_state.pending_query or st.chat_input(
    "Ask about any stock or company...",
    disabled=not keys_ok
)
if st.session_state.pending_query:
    st.session_state.pending_query = None

# =============================================================================
# PROCESS QUERY
# =============================================================================
if prompt:
    if not keys_ok:
        st.error("Please enter your API keys in the sidebar first.")
    elif st.session_state.agent is None:
        st.error("Agent not initialized — please check your API keys.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate charts for relevant queries
        charts_to_show = []
        tickers = extract_tickers(prompt.upper())

        if should_show_chart(prompt) and tickers:
            if len(tickers) == 1:
                charts_to_show.append({"type": "single", "ticker": tickers[0], "period": "1y"})
            elif len(tickers) > 1:
                charts_to_show.append({"type": "comparison", "tickers": tickers, "period": "1y"})

        # Agent response
        with st.chat_message("assistant"):

            # Show charts first
            for chart_data in charts_to_show:
                if chart_data["type"] == "single":
                    fig = plot_stock_chart(chart_data["ticker"], chart_data["period"])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                elif chart_data["type"] == "comparison":
                    fig = plot_comparison_chart(chart_data["tickers"], chart_data["period"])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

            # Agent analysis
            with st.spinner("🔍 Analyzing... (30-60 seconds)"):
                try:
                    result = st.session_state.agent.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config={"configurable": {"thread_id": st.session_state.thread_id}}
                    )
                    response = result["messages"][-1].content
                    st.markdown(response)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "charts": charts_to_show
                    })

                except Exception as e:
                    err = f"❌ Error: {str(e)}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})

# =============================================================================
# EMPTY STATE
# =============================================================================
if not st.session_state.messages and not (selected_ticker and selected_ticker != "Select..."):
    st.markdown("### Get Started")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **📈 Single Stock Analysis**
        - Analyze Apple (AAPL)
        - Tesla risk assessment
        - What's happening with NVDA?
        """)
    with c2:
        st.markdown("""
        **⚖️ Comparisons**
        - Compare NVIDIA vs AMD
        - Microsoft vs Google AI strategy
        - AMZN vs MSFT cloud
        """)
    with c3:
        st.markdown("""
        **🏆 Rankings**
        - Rank MSFT, GOOGL, NVDA, AMZN
        - Best AI stock to buy now?
        - Which company has best AI research?
        """)

    st.info("👈 Enter your API keys in the sidebar to start. Both have free tiers.")
