import { useState } from "react";
import { PenSquare, Trash2, Edit2, Check, X, LogOut, LogIn, UserPlus, ChevronLeft, ChevronRight, MessageSquare } from "lucide-react";
export default function Sidebar({ open, onToggle, chats, activeChatId, onNewChat, onSelectChat, onDeleteChat, onRenameChat, user, onLogout, onLogin, onRegister }) {
  const [editingId, setEditingId] = useState(null);
  const [editVal, setEditVal] = useState("");
  const startEdit = (c) => { setEditingId(c.id); setEditVal(c.title); };
  const commitEdit = (id) => { if (editVal.trim()) onRenameChat(id, editVal.trim()); setEditingId(null); };
  return (
    <>
      <div className={`sidebar ${open ? "open" : "closed"}`}>
        <div className="sidebar-header">
          <span className="sidebar-logo">🧠 LLM-Mini</span>
          <button className="icon-btn" onClick={onNewChat} title="New chat"><PenSquare size={16} /></button>
        </div>
        <div className="chat-list">
          {chats.length === 0 && <div className="no-chats">No chats yet.<br/>Start a new one!</div>}
          {chats.map(c => (
            <div key={c.id} className={`chat-item ${c.id === activeChatId ? "active" : ""}`} onClick={() => onSelectChat(c.id)}>
              <MessageSquare size={14} className="chat-icon" />
              {editingId === c.id ? (
                <input className="rename-input" value={editVal} onChange={e => setEditVal(e.target.value)}
                  onKeyDown={e => { if (e.key==="Enter") commitEdit(c.id); if (e.key==="Escape") setEditingId(null); }}
                  autoFocus onClick={e => e.stopPropagation()} />
              ) : (
                <span className="chat-title">{c.title}</span>
              )}
              <div className="chat-actions" onClick={e => e.stopPropagation()}>
                {editingId === c.id ? (
                  <><button className="icon-btn sm" onClick={() => commitEdit(c.id)}><Check size={12}/></button>
                  <button className="icon-btn sm" onClick={() => setEditingId(null)}><X size={12}/></button></>
                ) : (
                  <><button className="icon-btn sm" onClick={() => startEdit(c)}><Edit2 size={12}/></button>
                  <button className="icon-btn sm danger" onClick={() => onDeleteChat(c.id)}><Trash2 size={12}/></button></>
                )}
              </div>
            </div>
          ))}
        </div>
        <div className="sidebar-footer">
          {user ? (
            <>
              <div className="user-info">
                <div className="avatar">{user.username?.[0]?.toUpperCase() || "U"}</div>
                <span className="username">{user.username}</span>
              </div>
              <button className="icon-btn danger" onClick={onLogout} title="Logout"><LogOut size={16}/></button>
            </>
          ) : (
            <div className="auth-btns">
              <button className="auth-btn" onClick={onLogin}><LogIn size={14}/> Login</button>
              <button className="auth-btn primary" onClick={onRegister}><UserPlus size={14}/> Register</button>
            </div>
          )}
        </div>
      </div>
      <button className="sidebar-toggle" onClick={onToggle}>
        {open ? <ChevronLeft size={16}/> : <ChevronRight size={16}/>}
      </button>
    </>
  );
}
