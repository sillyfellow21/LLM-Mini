const TASK_LABELS = {
  farmer: { emoji: "🌾", label: "Agriculture" },
  story:  { emoji: "📖", label: "Story" },
  poetry: { emoji: "🎵", label: "Poetry" },
  qa:     { emoji: "❓", label: "Q&A" },
};
export default function MessageBubble({ msg }) {
  const isUser = msg.role === "user";
  const taskInfo = TASK_LABELS[msg.task];
  return (
    <div className={`message-row ${isUser ? "user" : "assistant"}`}>
      {!isUser && <div className="msg-avatar">🧠</div>}
      <div className={`bubble ${isUser ? "user-bubble" : "assistant-bubble"} ${msg.error ? "error-bubble" : ""}`}>
        {!isUser && taskInfo && (
          <div className="task-badge">
            {taskInfo.emoji} {taskInfo.label}
            {msg.used_wiki && <span className="wiki-badge">📚 Wikipedia</span>}
          </div>
        )}
        <p className="bubble-text">
          {msg.content}
          {msg.streaming && <span className="cursor">▍</span>}
        </p>
      </div>
      {isUser && <div className="msg-avatar user-av">U</div>}
    </div>
  );
}
