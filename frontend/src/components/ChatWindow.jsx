import { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble";
export default function ChatWindow({ messages }) {
  const bottomRef = useRef(null);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);
  return (
    <div className="chat-window">
      {messages.length === 0 && (
        <div className="empty-state">
          <div className="empty-icon">🧠</div>
          <h2>MiniLLM</h2>
          <p>Your personal AI assistant. Ask me anything!</p>
          <div className="hint-pills">
            <span>🌾 Farming tips</span>
            <span>📖 Tell me a story</span>
            <span>🎵 Write a poem</span>
            <span>❓ General Q&A</span>
          </div>
        </div>
      )}
      {messages.map(m => <MessageBubble key={m.id} msg={m} />)}
      <div ref={bottomRef} />
    </div>
  );
}
