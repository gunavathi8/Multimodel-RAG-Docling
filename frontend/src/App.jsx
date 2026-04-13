import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function App() {
  const [pdfFile, setPdfFile] = useState(null);
  const [ingestLoading, setIngestLoading] = useState(false);
  const [ingestResult, setIngestResult] = useState(null);
  const [ingestError, setIngestError] = useState("");

  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState("");
  const [messages, setMessages] = useState([]);
  const [latestChunks, setLatestChunks] = useState([]);

  const canSend = useMemo(
    () => query.trim().length > 0 && !chatLoading,
    [query, chatLoading]
  );

  const canIngest = useMemo(
    () => !!pdfFile && !ingestLoading,
    [pdfFile, ingestLoading]
  );

  async function handleIngest(e) {
    e.preventDefault();
    if (!pdfFile) return;

    setIngestLoading(true);
    setIngestError("");
    setIngestResult(null);

    try {
      const formData = new FormData();
      formData.append("file", pdfFile);

      const res = await fetch(`${API_BASE_URL}/ingest/pdf`, {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Ingestion failed");
      }

      setIngestResult(data);
    } catch (err) {
      setIngestError(err.message || "Ingestion failed");
    } finally {
      setIngestLoading(false);
    }
  }

  async function handleSendQuery(e) {
    e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;

    setChatLoading(true);
    setChatError("");

    const userMessage = { role: "user", content: trimmed };
    setMessages((prev) => [...prev, userMessage]);
    setQuery("");

    try {
      const res = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          query: trimmed,
          top_k: Number(topK),
          generate_answer: true
        })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Query failed");
      }

      setLatestChunks(data.chunks || []);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            data.answer && data.answer.trim().length > 0
              ? data.answer
              : "No answer generated."
        }
      ]);
    } catch (err) {
      setChatError(err.message || "Query failed");
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "The request failed. Please check backend logs and try again."
        }
      ]);
    } finally {
      setChatLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="control-panel">
        <h1>Multimodal RAG</h1>
        <p className="subtle">Docling + Pinecone + OpenRouter</p>

        <section className="panel-card">
          <h2>Ingestion</h2>
          <form onSubmit={handleIngest} className="stack">
            <label className="file-label" htmlFor="pdf-file">
              Select PDF
            </label>
            <input
              id="pdf-file"
              type="file"
              accept="application/pdf"
              onChange={(e) => setPdfFile(e.target.files?.[0] || null)}
            />
            <button type="submit" disabled={!canIngest}>
              {ingestLoading ? "Ingesting..." : "Ingest PDF"}
            </button>
          </form>
          {ingestError ? <p className="error-text">{ingestError}</p> : null}
          {ingestResult ? (
            <div className="metrics">
              <div>File: {ingestResult.file}</div>
              <div>Text chunks: {ingestResult.text_chunks}</div>
              <div>Figure chunks: {ingestResult.figure_chunks}</div>
              <div>Total chunks: {ingestResult.total_chunks}</div>
              <div>Index vectors: {ingestResult.index_total_vectors}</div>
            </div>
          ) : null}
        </section>

        <section className="panel-card">
          <h2>Retrieval</h2>
          <label htmlFor="topk">Top K</label>
          <input
            id="topk"
            type="number"
            min="1"
            max="20"
            value={topK}
            onChange={(e) => setTopK(e.target.value)}
          />
          <p className="subtle">API: {API_BASE_URL}</p>
        </section>

        <section className="panel-card">
          <h2>Latest Sources</h2>
          <div className="sources-list">
            {latestChunks.length === 0 ? (
              <p className="subtle">No query results yet.</p>
            ) : (
              latestChunks.map((chunk, idx) => (
                <article key={`${chunk.chunk_id}-${idx}`} className="source-card">
                  <div className="source-meta">
                    <span>#{idx + 1}</span>
                    <span>{chunk.chunk_type}</span>
                    <span>score {chunk.score}</span>
                  </div>
                  <div className="markdown-block source-markdown">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {chunk.text || ""}
                    </ReactMarkdown>
                  </div>
                </article>
              ))
            )}
          </div>
        </section>
      </aside>

      <main className="chat-panel">
        <div className="chat-header">
          <h2>Chat</h2>
          <p>Ask questions after ingestion completes.</p>
        </div>

        <div className="chat-history">
          {messages.length === 0 ? (
            <div className="empty-state">
              <p>Start by ingesting a PDF, then ask your first question.</p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div
                key={`${msg.role}-${idx}`}
                className={`message-row ${msg.role === "user" ? "from-user" : "from-assistant"}`}
              >
                <div className="message-bubble">
                  <div className="message-role">{msg.role === "user" ? "You" : "Assistant"}</div>
                  {msg.role === "assistant" ? (
                    <div className="markdown-block">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <p>{msg.content}</p>
                  )}
                </div>
              </div>
            ))
          )}
          {chatLoading ? (
            <div className="message-row from-assistant">
              <div className="message-bubble loading-bubble">
                <div className="message-role">Assistant</div>
                <p>Thinking...</p>
              </div>
            </div>
          ) : null}
        </div>

        <form className="chat-input" onSubmit={handleSendQuery}>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask about tables, figures, formulas, or summaries..."
            rows={3}
          />
          <button type="submit" disabled={!canSend}>
            Send
          </button>
        </form>
        {chatError ? <p className="error-text">{chatError}</p> : null}
      </main>
    </div>
  );
}

export default App;
