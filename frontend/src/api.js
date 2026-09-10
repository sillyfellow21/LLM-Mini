const configuredBase = import.meta.env.VITE_API_URL?.trim().replace(/\/$/, "");
export const API_BASE = configuredBase || (import.meta.env.DEV ? "/api" : "");
const requireApiBase = () => {
  if (!API_BASE) throw new Error("Backend URL is not configured. Set VITE_API_URL in Vercel.");
  return API_BASE;
};
const headers = () => {
  const t = localStorage.getItem("access_token");
  return t ? { Authorization: `Bearer ${t}` } : {};
};
export const api = {
  post: async (path, body) => {
    const res = await fetch(`${requireApiBase()}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...headers() },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    return data;
  },
  get: async (path) => {
    const res = await fetch(`${requireApiBase()}${path}`, { headers: headers() });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    return data;
  },
  put: async (path, body) => {
    const res = await fetch(`${requireApiBase()}${path}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json", ...headers() },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    return data;
  },
  del: async (path) => {
    const res = await fetch(`${requireApiBase()}${path}`, { method: "DELETE", headers: headers() });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    return data;
  },
};
