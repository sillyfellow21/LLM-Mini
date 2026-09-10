import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import Sidebar    from "../components/Sidebar";
import ChatWindow from "../components/ChatWindow";
import InputBar   from "../components/InputBar";
import { api }    from "../api";

const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function Chat({ guest = false }) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [streaming, setStreaming] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const esRef = useRef(null);

  useEffect(() => {
    if (user) api.get("/chats").then(setChats).catch(() => {});
  }, [user]);

  useEffect(() => {
    if (activeChatId && user) {
      api.get(`/chats/${activeChatId}/messages`).then(setMessages).catch(() => {});
    } else if (!activeChatId) { setMessages([]); }
  }, [activeChatId, user]);

  const newChat = async () => {
    if (!user) { setActiveChatId(null); setMessages([]); return; }
    const c = await api.post("/chats", { title: "New Chat" });
    setChats(prev => [c, ...prev]);
    setActiveChatId(c.id);
    setMessages([]);
  };

  const deleteChat = async (id) => {
    await api.del(`/chats/${id}`);
    setChats(prev => prev.filter(c => c.id !== id));
    if (activeChatId === id) { setActiveChatId(null); setMessages([]); }
  };

  const renameChat = async (id, title) => {
    await api.put(`/chats/${id}`, { title });
    setChats(prev => prev.map(c => c.id === id ? { ...c, title } : c));
  };

  const sendMessage = async (text) => {
    if (!text.trim() || streaming) return;
    let chatId = activeChatId;
    if (!chatId && user) {
      const c = await api.post("/chats", { title: "New Chat" });
      setChats(prev => [c, ...prev]);
      setActiveChatId(c.id);
      chatId = c.id;
    }
    const userMsg = { id: Date.now(),     role: "user",      content: text };
    const asstMsg = { id: Date.now() + 1, role: "assistant", content: "", task: null, used_wiki: false, streaming: true };
    setMessages(prev => [...prev, userMsg, asstMsg]);
    setStreaming(true);

    const token = localStorage.getItem("access_token");
    const params = new URLSearchParams({ message: text });
    if (chatId) params.append("chat_id", chatId);
    if (token)  params.append("token", token);

    if (esRef.current) esRef.current.close();
    const es = new EventSource(`${BASE}/chat/stream?${params}`);
    esRef.current = es;

    es.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type === "task") {
        setMessages(prev => prev.map(m => m.streaming ? { ...m, task: data.task, used_wiki: data.used_wiki } : m));
      } else if (data.type === "token") {
        setMessages(prev => prev.map(m => m.streaming ? { ...m, content: m.content + data.token } : m));
      } else if (data.type === "done") {
        setMessages(prev => prev.map(m => m.streaming ? { ...m, streaming: false } : m));
        setStreaming(false); es.close();
        if (user) api.get("/chats").then(setChats).catch(() => {});
      } else if (data.type === "error") {
        setMessages(prev => prev.map(m => m.streaming ? { ...m, content: data.message, streaming: false, error: true } : m));
        setStreaming(false); es.close();
      }
    };
    es.onerror = () => {
      setMessages(prev => prev.map(m => m.streaming ? { ...m, content: "Connection error. Is the backend running?", streaming: false, error: true } : m));
      setStreaming(false); es.close();
    };
  };

  return (
    <div className="app-layout">
      <Sidebar open={sidebarOpen} onToggle={() => setSidebarOpen(p => !p)}
        chats={chats} activeChatId={activeChatId} onNewChat={newChat}
        onSelectChat={setActiveChatId} onDeleteChat={deleteChat} onRenameChat={renameChat}
        user={user} onLogout={() => { logout(); navigate("/login"); }}
        onLogin={() => navigate("/login")} onRegister={() => navigate("/register")} />
      <div className={`main-area ${sidebarOpen ? "sidebar-open" : ""}`}>
        {!user && (
          <div className="guest-banner">
            👋 Guest mode (50 message limit) —{" "}
            <span className="link" onClick={() => navigate("/register")}>Register free</span> for unlimited access
          </div>
        )}
        <ChatWindow messages={messages} streaming={streaming} />
        <InputBar onSend={sendMessage} disabled={streaming} />
      </div>
    </div>
  );
}
