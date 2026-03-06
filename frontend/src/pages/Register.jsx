import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
export default function Register() {
  const [form, setForm] = useState({ email: "", username: "", password: "" });
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(false);
  const update = k => e => setForm(f => ({ ...f, [k]: e.target.value }));
  const submit = async (e) => {
    e.preventDefault(); setError(""); setLoading(true);
    try { const r = await api.post("/auth/register", form); setSuccess(r.message); }
    catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };
  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-logo">🧠</div>
        <h1>Create account</h1>
        <p className="auth-sub">Join MiniLLM for unlimited access</p>
        {error && <div className="alert error">{error}</div>}
        {success && <div className="alert success">{success}</div>}
        {!success && (
          <form onSubmit={submit}>
            <label>Email</label>
            <input type="email" value={form.email} onChange={update("email")} required placeholder="you@example.com" />
            <label>Username</label>
            <input type="text" value={form.username} onChange={update("username")} required placeholder="cooluser" />
            <label>Password</label>
            <input type="password" value={form.password} onChange={update("password")} required placeholder="Min 8 characters" />
            <button className="auth-submit" disabled={loading}>{loading ? "Creating…" : "Create account"}</button>
          </form>
        )}
        <p className="auth-switch">Already have an account? <Link to="/login">Login</Link></p>
      </div>
    </div>
  );
}
