import { useEffect, useState } from "react";
import { useSearchParams, Link } from "react-router-dom";
import { api } from "../api";
export default function VerifyEmail() {
  const [params] = useSearchParams();
  const [status, setStatus] = useState("verifying");
  const [message, setMessage] = useState("");
  useEffect(() => {
    const token = params.get("token");
    if (!token) { setStatus("error"); setMessage("No token found."); return; }
    api.post("/auth/verify-email", { token })
      .then(d => { setStatus("success"); setMessage(d.message); })
      .catch(e => { setStatus("error"); setMessage(e.message); });
  }, []);
  return (
    <div className="auth-page">
      <div className="auth-card centered">
        {status === "verifying" && <><div className="spinner"/><p>Verifying…</p></>}
        {status === "success" && <><div className="big-icon">✅</div><h2>Verified!</h2><p>{message}</p><Link className="auth-submit btn-link" to="/login">Go to Login</Link></>}
        {status === "error" && <><div className="big-icon">❌</div><h2>Failed</h2><p>{message}</p><Link className="auth-submit btn-link" to="/register">Back to Register</Link></>}
      </div>
    </div>
  );
}
