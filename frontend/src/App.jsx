import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const SAMPLE_QUESTIONS = [
  "How do I upload a document?",
  "How do I delete multiple documents?",
  "What are relationships in Revealr?",
  "Why do we need to add levels?",
  "How do I use section relationships step by step?"
];

function App() {
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [pdfFile, setPdfFile] = useState(null);
  const [llmProvider, setLlmProvider] = useState("openrouter");
  const [ingestLoading, setIngestLoading] = useState(false);
  const [ingestResult, setIngestResult] = useState(null);
  const [ingestError, setIngestError] = useState("");

  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [useDynamicRetrieval, setUseDynamicRetrieval] = useState(true);
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
      formData.append("llm_provider", llmProvider);

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
    const historyPayload = messages
      .filter((m) => m.role === "user" || m.role === "assistant")
      .slice(-5)
      .map((m) => ({
        role: m.role,
        content: m.content
      }));
    const assistantIndexRef = { value: -1 };
    setMessages((prev) => {
      const next = [
        ...prev,
        userMessage,
        { role: "assistant", content: "" }
      ];
      assistantIndexRef.value = next.length - 1;
      return next;
    });
    setQuery("");

    try {
      const res = await fetch(`${API_BASE_URL}/query/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          query: trimmed,
          top_k: Number(topK),
          generate_answer: true,
          llm_provider: llmProvider,
          use_dynamic_retrieval: useDynamicRetrieval,
          chat_history: historyPayload
        })
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Query failed");
      }

      const reader = res.body?.getReader();
      if (!reader) {
        throw new Error("Streaming not supported by browser.");
      }

      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let assistantText = "";
      let receivedDone = false;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() || "";

        for (const eventBlock of events) {
          const line = eventBlock
            .split("\n")
            .find((l) => l.startsWith("data: "));
          if (!line) continue;

          const payload = JSON.parse(line.slice(6));
          if (payload.type === "meta") {
            setLatestChunks(payload.chunks || []);
          } else if (payload.type === "token") {
            assistantText += payload.token || "";
            setMessages((prev) =>
              prev.map((m, idx) =>
                idx === assistantIndexRef.value
                  ? { ...m, content: assistantText }
                  : m
              )
            );
          } else if (payload.type === "done") {
            receivedDone = true;
          } else if (payload.type === "error") {
            throw new Error(payload.message || "Streaming failed");
          }
        }
      }

      if (!receivedDone && assistantText.trim().length === 0) {
        setMessages((prev) =>
          prev.map((m, idx) =>
            idx === assistantIndexRef.value
              ? { ...m, content: "No answer generated." }
              : m
          )
        );
      }
    } catch (err) {
      setChatError(err.message || "Query failed");
      setMessages((prev) =>
        prev.map((m, idx) =>
          m.role === "assistant" && idx === prev.length - 1
            ? { ...m, content: "The request failed. Please check backend logs and try again." }
            : m
        )
      );
    } finally {
      setChatLoading(false);
    }
  }

  function handleSampleQuestionClick(sampleQuestion) {
    setQuery(sampleQuestion);
  }

  return (
    <div className={`app-shell ${isSidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      <aside className={`control-panel ${isSidebarCollapsed ? "collapsed" : ""}`}>
        <button
          type="button"
          className="sidebar-toggle-btn"
          onClick={() => setIsSidebarCollapsed((prev) => !prev)}
          aria-label={isSidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          title={isSidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {isSidebarCollapsed ? "»" : "«"}
        </button>

        {isSidebarCollapsed ? (
          <div className="collapsed-label">Controls</div>
        ) : null}

        {!isSidebarCollapsed ? (
          <>
        <h1>Multimodal RAG</h1>
        <p className="subtle">Docling + Pinecone + OpenRouter/Azure OpenAI</p>

        <section className="panel-card">
          <h2>Ingestion</h2>
          <form onSubmit={handleIngest} className="stack">
            <label htmlFor="provider">LLM Provider</label>
            <select
              id="provider"
              value={llmProvider}
              onChange={(e) => setLlmProvider(e.target.value)}
            >
              <option value="openrouter">OpenRouter</option>
              <option value="azure">Azure OpenAI</option>
            </select>
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
              <div>Provider: {ingestResult.provider}</div>
              <div>Index: {ingestResult.index_name}</div>
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
          <label className="checkbox-label" htmlFor="dynamic-retrieval">
            <input
              id="dynamic-retrieval"
              type="checkbox"
              checked={useDynamicRetrieval}
              onChange={(e) => setUseDynamicRetrieval(e.target.checked)}
            />
            Use dynamic chunk retrieval (recommended)
          </label>
          <p className="subtle">API: {API_BASE_URL}</p>
          <p className="subtle">Selected provider: {llmProvider}</p>
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
          </>
        ) : null}
      </aside>

      <main className="chat-panel">
        <div className="chat-header">
          <h2>Chat</h2>
          <p>Ask questions after ingestion completes.</p>
        </div>

        <div className="chat-history">
          <div className="sample-questions">
            {SAMPLE_QUESTIONS.map((sampleQuestion) => (
              <button
                key={sampleQuestion}
                type="button"
                className="sample-question-btn"
                onClick={() => handleSampleQuestionClick(sampleQuestion)}
                disabled={chatLoading}
              >
                {sampleQuestion}
              </button>
            ))}
          </div>
          {messages.length === 0 ? (
            <div className="empty-state" />
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
        </div>

        <form className="chat-input" onSubmit={handleSendQuery}>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask anything about Revealr.ai..."
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
