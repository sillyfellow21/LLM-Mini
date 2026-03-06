import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPass] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const submit = async (e) => {
    e.preventDefault(); setError(""); setLoading(true);
    try { await login(email, password); navigate("/"); }
    catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };
  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-logo">🧠</div>
        <h1>Welcome back</h1>
        <p className="auth-sub">Sign in to your MiniLLM account</p>
        {error && <div className="alert error">{error}</div>}
        <form onSubmit={submit}>
          <label>Email</label>
          <input type="email" value={email} onChange={e => setEmail(e.target.value)} required placeholder="you@example.com" />
          <label>Password</label>
          <input type="password" value={password} onChange={e => setPass(e.target.value)} required placeholder="••••••••" />
          <div className="auth-extra"><Link to="/forgot-password">Forgot password?</Link></div>
          <button className="auth-submit" disabled={loading}>{loading ? "Signing in…" : "Sign in"}</button>
        </form>
        <p className="auth-switch">Don't have an account? <Link to="/register">Register</Link></p>
        <p className="auth-switch"><Link to="/">Continue as guest →</Link></p>
      </div>
    </div>
  );
}
