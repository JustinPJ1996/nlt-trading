import { useState, type FormEvent } from "react";
import { api, ApiError } from "../api";
import { ErrorLine } from "../components/Callout";

export function Login({ onDone }: { onDone: (user: string) => void }) {
  const [id, setId] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const r = await api.login(id, password);
      onDone(r.user);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not log in.");
      setPassword("");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-page">
      <form className="login-card panel" onSubmit={submit}>
        <h1 className="wordmark">Strategy Desk</h1>
        <p className="muted" style={{ marginBottom: 24 }}>
          Log in to build and test your strategies.
        </p>
        <label className="field">
          <span className="label">Login ID</span>
          <input type="text" autoComplete="username" value={id} onChange={(e) => setId(e.target.value)} required autoFocus />
        </label>
        <label className="field">
          <span className="label">Password</span>
          <input type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} required />
        </label>
        <div style={{ marginTop: 20 }} className="stack">
          <ErrorLine error={error} />
          <button className="btn" type="submit" disabled={busy} style={{ width: "100%", justifyContent: "center" }}>
            {busy ? "Logging in..." : "Log in"}
          </button>
        </div>
      </form>
    </div>
  );
}
