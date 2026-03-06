import { useAuth } from "../context/AuthContext";
export default function ProtectedRoute({ children, fallback }) {
  const { user, loading } = useAuth();
  if (loading) return <div className="loading-screen">Loading…</div>;
  if (!user) return fallback || null;
  return children;
}
