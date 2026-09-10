import { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble";
const suggestions = [
  { label: "🌾 Farming tips", prompt: "Give me three practical farming tips." },
  { label: "📖 Tell me a story", prompt: "Tell me a short story." },
  { label: "🎵 Write a poem", prompt: "Write me a short poem." },
  { label: "❓ General Q&A", prompt: "What are some interesting facts I should know?" },
];

export default function ChatWindow({ messages, onSuggestion, disabled }) {
  const bottomRef = useRef(null);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);
  return (
    <div className="chat-window">
      {messages.length === 0 && (
        <div className="empty-state">
          <div className="empty-icon">🧠</div>
          <h2>LLM-Mini</h2>
          <p>Your personal AI assistant. Ask me anything!</p>
          <div className="hint-pills">
            {suggestions.map(({ label, prompt }) => (
              <button
                className="hint-pill"
                key={label}
                type="button"
                onClick={() => onSuggestion(prompt)}
                disabled={disabled}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      )}
      {messages.map(m => <MessageBubble key={m.id} msg={m} />)}
      <div ref={bottomRef} />
    </div>
  );
}
