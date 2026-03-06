import { createContext, useContext, useState, useEffect } from "react";
import { api } from "../api";
const Ctx = createContext(null);
export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    const token = localStorage.getItem("access_token");
    if (token) {
      api.get("/user/profile").then(setUser).catch(() => localStorage.removeItem("access_token")).finally(() => setLoading(false));
    } else { setLoading(false); }
  }, []);
  const login = async (email, password) => {
    const data = await api.post("/auth/login", { email, password });
    localStorage.setItem("access_token", data.access_token);
    localStorage.setItem("refresh_token", data.refresh_token);
    setUser(data.user);
  };
  const logout = () => { localStorage.removeItem("access_token"); localStorage.removeItem("refresh_token"); setUser(null); };
  return <Ctx.Provider value={{ user, login, logout, loading }}>{children}</Ctx.Provider>;
}
export const useAuth = () => useContext(Ctx);
