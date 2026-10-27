"use client";
import { useEffect, useState } from "react";
const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
export default function History() {
  const [rows, setRows] = useState<{prediction:string; confidence:number; created_at:string}[]>([]);
  useEffect(() => { fetch(`${API}/api/v1/predictions`).then(r => r.ok ? r.json() : []).then(setRows).catch(() => setRows([])); }, []);
  return <main className="mx-auto max-w-3xl px-5 py-12"><a className="text-teal" href="/">← Dashboard</a><h1 className="mt-8 text-4xl font-black">Prediction history</h1><p className="mt-3 text-slate-600">Recent scores are stored only when PostgreSQL is configured.</p><div className="card mt-8 overflow-x-auto"><table className="w-full text-left text-sm"><thead><tr className="border-b"><th className="pb-3">Time</th><th className="pb-3">Outcome</th><th className="pb-3">Confidence</th></tr></thead><tbody>{rows.map((row, i) => <tr key={i} className="border-b last:border-0"><td className="py-3">{new Date(row.created_at).toLocaleString()}</td><td className="py-3 font-semibold">{row.prediction}</td><td className="py-3">{(row.confidence * 100).toFixed(1)}%</td></tr>)}</tbody></table>{!rows.length && <p className="py-5 text-slate-500">No saved predictions yet.</p>}</div></main>;
}
