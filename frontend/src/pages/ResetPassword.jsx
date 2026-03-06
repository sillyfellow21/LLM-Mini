import { useState } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { api } from "../api";
export default function ResetPassword() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const [password, setPass] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const submit = async (e) => {
    e.preventDefault(); setError(""); setLoading(true);
    try { await api.post("/auth/reset-password", { token: params.get("token"), password }); navigate("/login"); }
    catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };
  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-logo">🔒</div>
        <h1>New password</h1>
        {error && <div className="alert error">{error}</div>}
        <form onSubmit={submit}>
          <label>New Password</label>
          <input type="password" value={password} onChange={e => setPass(e.target.value)} required placeholder="Min 8 characters" />
          <button className="auth-submit" disabled={loading}>{loading ? "Saving…" : "Reset password"}</button>
        </form>
      </div>
    </div>
  );
}
