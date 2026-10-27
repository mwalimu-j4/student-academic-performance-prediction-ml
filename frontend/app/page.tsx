"use client";

import { FormEvent, useMemo, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const fields = [
  ["Marital status", 1], ["Application mode", 17], ["Application order", 0],
  ["Course", 171], ["Daytime/evening attendance", 1], ["Previous qualification", 1],
  ["Previous qualification (grade)", 130], ["Nacionality", 1],
  ["Mother's qualification", 19], ["Father's qualification", 12],
  ["Mother's occupation", 5], ["Father's occupation", 9], ["Admission grade", 127],
  ["Displaced", 1], ["Educational special needs", 0], ["Debtor", 0],
  ["Tuition fees up to date", 1], ["Gender", 1], ["Scholarship holder", 0],
  ["Age at enrollment", 20], ["International", 0], ["Unemployment rate", 10.8],
  ["Inflation rate", 1.4], ["GDP", 1.74],
] as const;

type Result = { prediction: string; confidence: number; probabilities: Record<string, number> };

export default function Home() {
  const initial = useMemo(() => Object.fromEntries(fields.map(([name, value]) => [name, value])), []);
  const [values, setValues] = useState<Record<string, number>>(initial);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  async function submit(event: FormEvent) {
    event.preventDefault(); setLoading(true); setError(""); setResult(null);
    try {
      const response = await fetch(`${API}/api/v1/predict`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: values }),
      });
      if (!response.ok) throw new Error((await response.json()).detail || "Prediction failed");
      setResult(await response.json());
    } catch (reason) { setError(reason instanceof Error ? reason.message : "API unavailable"); }
    finally { setLoading(false); }
  }
  return <main className="mx-auto max-w-7xl px-5 py-8">
    <header className="mb-8 flex flex-wrap items-end justify-between gap-4">
      <div><p className="mb-2 text-sm font-bold uppercase tracking-[.2em] text-teal">Student Outcome Intelligence</p>
        <h1 className="text-4xl font-black tracking-tight md:text-5xl">A clearer early signal for student support.</h1>
        <p className="mt-3 max-w-2xl text-slate-600">An admission-time, three-class model for Dropout, Enrolled, and Graduate outcomes. Predictions are decision support, never a verdict.</p></div>
      <div className="rounded-full bg-teal/10 px-4 py-2 text-sm font-semibold text-teal">UCI 697 · 4,424 records</div>
    </header>
    <section className="grid gap-6 lg:grid-cols-[1.4fr_.8fr]">
      <form className="card" onSubmit={submit}><div className="mb-5 flex items-center justify-between"><h2 className="text-xl font-bold">Admission profile</h2><span className="text-sm text-slate-500">24 validated fields</span></div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">{fields.map(([name, defaultValue]) =>
          <label className="label" key={name}>{name}<input className="input mt-1" type="number" step="any" value={values[name]} required onChange={e => setValues({...values, [name]: Number(e.target.value)})} aria-label={name} />
        </label>)}</div>
        <button disabled={loading} className="mt-6 w-full rounded-lg bg-ink px-5 py-3 font-bold text-white transition hover:bg-teal disabled:opacity-50">{loading ? "Scoring profile…" : "Predict outcome"}</button>
        {error && <p className="mt-4 rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</p>}
      </form>
      <aside className="space-y-6"><div className="card min-h-56">{result ? <><p className="label">Model result</p><h2 className="mt-2 text-4xl font-black text-teal">{result.prediction}</h2><p className="mt-2 text-slate-600">Estimated confidence: <strong>{(result.confidence * 100).toFixed(1)}%</strong></p><div className="mt-6 space-y-3">{Object.entries(result.probabilities).map(([label, probability]) => <div key={label}><div className="flex justify-between text-sm"><span>{label}</span><span>{(probability * 100).toFixed(1)}%</span></div><div className="mt-1 h-2 rounded-full bg-slate-100"><div className="h-2 rounded-full bg-teal" style={{width: `${probability * 100}%`}} /></div></div>)}</div></> : <><p className="label">Your result</p><h2 className="mt-2 text-2xl font-bold">Complete the profile to see a prediction.</h2><p className="mt-3 text-slate-600">The score is the model&apos;s estimated class probability, not a calibrated guarantee or individual judgment.</p></>}</div>
        <div className="card"><h2 className="font-bold">How to read this</h2><ul className="mt-3 list-disc space-y-2 pl-5 text-sm text-slate-600"><li>Inputs are restricted to information available at admission.</li><li>Curricular-unit progress and grades are excluded to prevent leakage.</li><li>UCI data covers two Portuguese institutions; external performance is not guaranteed.</li></ul></div></aside>
    </section>
    <footer className="mt-8 border-t border-slate-200 pt-5 text-sm text-slate-500">Research dashboard · UCI Machine Learning Repository dataset 697 · CC BY 4.0</footer>
  </main>;
}
