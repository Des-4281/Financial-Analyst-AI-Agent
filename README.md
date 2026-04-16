# 📊 Autonomous Financial Research Analyst

An AI agent that autonomously gathers live stock data, real-time news, sentiment analysis, and private analyst reports to generate structured investment recommendations.

Built with **LangGraph**, **LangChain**, and **GPT-4o-mini**.

---

## Demo

> **You:** Analyze NVIDIA stock
>
> **Agent:** *calls get_stock_price, get_stock_history, search_financial_news, analyze_sentiment, query_private_database simultaneously*
>
> ### Investment Analysis — NVIDIA (NVDA)
> **Executive Summary:** NVIDIA has delivered a 594% return over 3 years...
> **Recommendation: BUY — 85% confidence**

---

## Quickstart

```bash
git clone https://github.com/your-username/financial-analyst-agent
cd financial-analyst-agent
pip install -r requirements.txt
cp config.example.json config.json
```

Open `config.json` and add your API keys, then run:

```bash
python agent.py
```

**API keys needed (both have free tiers):**
- OpenAI: https://platform.openai.com
- Tavily: https://app.tavily.com

---

## Example Queries

```
Analyze Apple stock (AAPL)
Compare NVIDIA and AMD for AI investment
Rank MSFT, GOOGL, NVDA, AMZN, IBM by investment potential
What are the risks in investing in Tesla right now?
Which company has the most innovative AI research?
```

---

## Features

- **Live stock data** — current price, volume, market cap via Yahoo Finance
- **3-year historical performance** — return %, highs, lows, average volume
- **Real-time news** — top 5 relevant articles via Tavily search
- **Sentiment scoring** — GPT-4o-mini scores each article 0.0 to 1.0
- **Private knowledge base** — optional RAG over your own PDF analyst reports
- **Structured reports** — executive summary, risk factors, Buy/Hold/Sell with confidence %
- **Conversation memory** — agent remembers context across turns in the same session
- **Graceful error handling** — if one tool fails, the agent continues with available data

---

## Architecture

```
Your Query
    ↓
Agent (GPT-4o-mini) decides which tools to call
    ↓
Tools run:
  get_stock_price()        → Yahoo Finance
  get_stock_history()      → Yahoo Finance
  search_financial_news()  → Tavily API
  analyze_sentiment()      → GPT-4o-mini
  query_private_database() → ChromaDB RAG (optional)
    ↓
Agent synthesizes all results
    ↓
Structured investment report with recommendation
```

---

## Optional: Private Knowledge Base

Drop PDF analyst reports into a `Companies-AI-Initiatives/` folder and the agent will automatically index them and use them to answer questions about company AI strategies, roadmaps, and initiatives.

```
financial-analyst-agent/
├── Companies-AI-Initiatives/
│   ├── MSFT.pdf
│   ├── NVDA.pdf
│   └── GOOGL.pdf
```

Or provide a `Companies-AI-Initiatives.zip` and it will extract automatically.

---

## Web Interface (Hugging Face Spaces)

A Streamlit UI version is available in `app.py`. To deploy on Hugging Face Spaces:

1. Create a new Space at huggingface.co (choose Streamlit SDK)
2. Upload `app.py` and `requirements_hf.txt` (rename to `requirements.txt`)
3. Add `OPENAI_API_KEY` and `TAVILY_API_KEY` in Space Settings → Secrets
4. Share the Space URL — users enter their own keys in the sidebar

---

## Project Structure

```
financial-analyst-agent/
├── agent.py               ← Terminal version (run this for CLI)
├── app.py                 ← Streamlit web UI (for Hugging Face)
├── requirements.txt       ← CLI dependencies
├── requirements_hf.txt    ← Web UI dependencies (includes streamlit)
├── config.example.json    ← Key format reference (copy to config.json)
├── .gitignore             ← Keeps your real keys off GitHub
└── README.md
```

---

## Roadmap

- [ ] Stock price charts and trend visualization
- [ ] Multiple data source fallbacks (Alpha Vantage, Polygon.io)
- [ ] Technical analysis (moving averages, RSI, candlestick patterns)
- [ ] Cross-stock correlation analysis
- [ ] Geopolitical risk scoring
- [ ] Institutional investor sentiment tracking
- [ ] Portfolio-level analysis and position sizing

---

## License

MIT — free to use, modify, and build on.
