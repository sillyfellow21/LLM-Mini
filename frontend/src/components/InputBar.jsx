import { useState } from "react";
import { Send } from "lucide-react";
export default function InputBar({ onSend, disabled }) {
  const [text, setText] = useState("");
  const submit = () => { if (!text.trim() || disabled) return; onSend(text.trim()); setText(""); };
  const onKey = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } };
  return (
    <div className="input-bar">
      <div className="input-wrap">
        <textarea className="chat-input" value={text} onChange={e => setText(e.target.value)}
          onKeyDown={onKey} placeholder="Message LLM-Mini…" rows={1} disabled={disabled} />
        <button className={`send-btn ${disabled || !text.trim() ? "disabled" : ""}`}
          onClick={submit} disabled={disabled || !text.trim()}>
          <Send size={16} />
        </button>
      </div>
      <p className="input-hint">Enter to send · Shift+Enter for new line</p>
    </div>
  );
}
