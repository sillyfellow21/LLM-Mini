import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
export default function ForgotPassword() {
  const [email, setEmail] = useState("");
  const [msg, setMsg] = useState("");
  const [loading, setLoading] = useState(false);
  const submit = async (e) => {
    e.preventDefault(); setLoading(true);
    try { const r = await api.post("/auth/forgot-password", { email }); setMsg(r.message); }
    catch (err) { setMsg(err.message); }
    finally { setLoading(false); }
  };
  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-logo">🔑</div>
        <h1>Reset password</h1>
        <p className="auth-sub">We'll send a reset link to your email.</p>
        {msg && <div className="alert success">{msg}</div>}
        {!msg && (
          <form onSubmit={submit}>
            <label>Email</label>
            <input type="email" value={email} onChange={e => setEmail(e.target.value)} required placeholder="you@example.com" />
            <button className="auth-submit" disabled={loading}>{loading ? "Sending…" : "Send reset link"}</button>
          </form>
        )}
        <p className="auth-switch"><Link to="/login">← Back to login</Link></p>
      </div>
    </div>
  );
}
