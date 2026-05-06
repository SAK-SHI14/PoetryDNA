import { motion } from "framer-motion";
import { useDeferredValue, useEffect, useRef, useState, useTransition } from "react";
import EmbeddingPlot3D from "../components/EmbeddingPlot3D";
import SimilarLines from "../components/SimilarLines";
import useScrollReveal from "../hooks/useScrollReveal";
import { explainPoetry, getCorpusEmbeddings } from "../lib/api";
import { SAMPLE_TEXT } from "../lib/poets";

function SbertArchitecture() {
  const steps = ["Input", "SBERT encoder", "Mean pooling", "384-d vector", "Cosine similarity", "Ranked lines"];
  return (
    <svg viewBox="0 0 960 220" className="w-full">
      {steps.map((step, index) => {
        const x = 24 + index * 154;
        return (
          <g key={step}>
            <rect
              x={x}
              y="76"
              width="126"
              height="64"
              rx="20"
              fill="rgba(255,255,255,0.04)"
              stroke="rgba(201,168,76,0.28)"
            />
            <text x={x + 63} y="114" textAnchor="middle" fill="#E8E0D0" style={{ fontFamily: "DM Mono", fontSize: "12px" }}>
              {step}
            </text>
            {index < steps.length - 1 ? (
              <path d={`M ${x + 126} 108 H ${x + 154}`} stroke="rgba(201,168,76,0.7)" strokeWidth="2" strokeDasharray="5 6" />
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}

export default function SBERT() {
  const containerRef = useRef(null);
  useScrollReveal(containerRef);

  const [text, setText] = useState(SAMPLE_TEXT.sbert);
  const deferredText = useDeferredValue(text);
  const [corpus, setCorpus] = useState([]);
  const [payload, setPayload] = useState(null);
  const [loadingCorpus, setLoadingCorpus] = useState(true);
  const [loadingExplain, setLoadingExplain] = useState(false);
  const [angle, setAngle] = useState(35);
  const [error, setError] = useState("");
  const [isPending, startTransition] = useTransition();

  useEffect(() => {
    let active = true;
    setLoadingCorpus(true);

    getCorpusEmbeddings()
      .then((response) => {
        if (active) {
          setCorpus(response.points ?? []);
        }
      })
      .catch((requestError) => {
        if (active) {
          setError(requestError.message);
        }
      })
      .finally(() => {
        if (active) {
          setLoadingCorpus(false);
        }
      });

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!deferredText.trim()) {
      return undefined;
    }

    let active = true;
    setLoadingExplain(true);
    setError("");

    const timer = window.setTimeout(async () => {
      try {
        const response = await explainPoetry(deferredText, ["sbert"]);
        if (!active) {
          return;
        }
        startTransition(() => setPayload(response));
      } catch (requestError) {
        if (active) {
          setError(requestError.message);
        }
      } finally {
        if (active) {
          setLoadingExplain(false);
        }
      }
    }, 360);

    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [deferredText, startTransition]);

  const sbert = payload?.sbert;
  const queryPoint = sbert?.query_projection_3d ?? null;
  const retrievedLines =
    sbert?.filtered_neighbors?.length > 0 ? sbert.filtered_neighbors : (sbert?.nearest_neighbors ?? []).slice(0, 3);
  const cosineValue = Math.cos((angle * Math.PI) / 180).toFixed(3);

  return (
    <main ref={containerRef} className="relative pb-28 pt-10 sm:pt-16">
      <div className="content-wrap space-y-14">
        <section data-reveal className="relative overflow-hidden rounded-[40px] border border-gold/10 p-10 sm:p-14">
          <div className="absolute inset-0 hero-glow opacity-60" />
          <div className="relative z-10 max-w-4xl">
            <p className="section-eyebrow text-xs">SBERT visual explanation</p>
            <h1 className="mt-5 font-display text-5xl uppercase tracking-[0.08em] text-text sm:text-6xl lg:text-[74px]">
              How the similarity brain finds kindred verses
            </h1>
            <p className="mt-6 max-w-3xl text-lg leading-8 text-text-dim">
              Twin encoders compress verse into a shared semantic space, then cosine geometry reveals which lines breathe in the same stylistic air.
            </p>
          </div>
        </section>

        <section data-reveal className="glass-panel rounded-[36px] p-8">
          <div className="grid gap-8 lg:grid-cols-[1.05fr_0.95fr]">
            <div>
              <p className="section-eyebrow text-xs">Dual encoding prompt</p>
              <textarea
                value={text}
                onChange={(event) => setText(event.target.value)}
                className="mt-5 min-h-[220px] w-full rounded-[28px] border border-gold/15 bg-black/25 p-6 text-lg leading-8 text-text outline-none transition focus:border-gold/45"
              />
            </div>
            <div className="space-y-4">
              <div className="metric-card rounded-[28px] p-6">
                <p className="section-eyebrow text-[10px]">Predicted poet filter</p>
                <p className="mt-3 font-display text-4xl text-gold">{sbert?.predicted_poet ?? "Scanning"}</p>
              </div>
              <div className="metric-card rounded-[28px] p-6">
                <p className="section-eyebrow text-[10px]">Corpus state</p>
                <p className="mt-3 font-mono text-sm uppercase tracking-[0.28em] text-text-dim">
                  {loadingCorpus ? "Loading embedding field" : `${corpus.length} points mapped`}
                </p>
              </div>
              <div className="metric-card rounded-[28px] p-6">
                <p className="section-eyebrow text-[10px]">Query state</p>
                <p className="mt-3 font-mono text-sm uppercase tracking-[0.28em] text-text-dim">
                  {loadingExplain ? "Projecting input" : isPending ? "Lighting neighbors" : "Ready"}
                </p>
              </div>
            </div>
          </div>
          {error ? (
            <p className="mt-6 rounded-2xl border border-crimson/20 bg-crimson/10 px-4 py-3 text-sm text-[#d7a3a3]">{error}</p>
          ) : null}
        </section>

        <section data-reveal className="grid gap-6 lg:grid-cols-2">
          <article className="glass-panel rounded-[32px] p-8">
            <p className="section-eyebrow text-xs">Step 1</p>
            <h2 className="mt-3 font-display text-3xl text-text">Dual Encoding</h2>
            <p className="mt-3 text-sm leading-7 text-text-dim">
              SBERT runs <em>both</em> texts through the same encoder simultaneously. Each bar below represents one dimension of the <span className="text-gold font-mono">384-d semantic vector</span>. Similar poems produce similar bar patterns — that similarity is how the system finds kindred lines.
            </p>
            <div className="mt-6 grid gap-4 md:grid-cols-2">
              <div className="rounded-[24px] border border-gold/15 bg-gold/5 p-5">
                <p className="font-display text-xl text-gold">Input poem</p>
                <p className="mt-1 font-mono text-[9px] uppercase tracking-widest text-text-dim">Your verse → 384 numbers</p>
                <div className="mt-4 space-y-1.5">
                  {Array.from({ length: 12 }, (_, index) => (
                    <motion.div
                      key={index}
                      initial={{ width: "20%" }}
                      animate={{ width: `${30 + ((index * 17) % 62)}%` }}
                      transition={{ repeat: Number.POSITIVE_INFINITY, repeatType: "reverse", duration: 2.8 + index * 0.15, delay: index * 0.09 }}
                      className="h-1.5 rounded-full"
                      style={{ background: `rgba(201,168,76,${0.4 + (index % 4) * 0.12})` }}
                    />
                  ))}
                </div>
                <p className="mt-3 font-mono text-[9px] text-text-dim">← semantic fingerprint</p>
              </div>
              <div className="rounded-[24px] border border-crimson/15 bg-crimson/5 p-5">
                <p className="font-display text-xl" style={{ color: "#d9a5a5" }}>Corpus line</p>
                <p className="mt-1 font-mono text-[9px] uppercase tracking-widest text-text-dim">Each stored poem → 384 numbers</p>
                <div className="mt-4 space-y-1.5">
                  {Array.from({ length: 12 }, (_, index) => (
                    <motion.div
                      key={index}
                      initial={{ width: "25%" }}
                      animate={{ width: `${28 + ((index * 13) % 60)}%` }}
                      transition={{ repeat: Number.POSITIVE_INFINITY, repeatType: "reverse", duration: 3.1 + index * 0.12, delay: index * 0.11 }}
                      className="h-1.5 rounded-full"
                      style={{ background: `rgba(139,26,26,${0.4 + (index % 4) * 0.15})` }}
                    />
                  ))}
                </div>
                <p className="mt-3 font-mono text-[9px] text-text-dim">← semantic fingerprint</p>
              </div>
            </div>
            <div className="mt-4 rounded-2xl border border-white/5 bg-white/[0.02] p-4">
              <p className="text-sm text-text-dim">When two poems discuss similar themes with similar rhythm, their 384-d vectors point in nearly the same direction in semantic space.</p>
            </div>
          </article>

          <article className="glass-panel rounded-[32px] p-8">
            <p className="section-eyebrow text-xs">Step 3</p>
            <h2 className="mt-3 font-display text-3xl text-text">Cosine Similarity</h2>
            <p className="mt-3 text-sm leading-7 text-text-dim">
              Drag the angle to see what different similarity scores mean. The smaller the angle between two vectors, the more alike the poems.
            </p>
            <div className="mt-6 space-y-5">
              <input
                type="range"
                min="0"
                max="180"
                value={angle}
                onChange={(event) => setAngle(Number(event.target.value))}
                className="w-full accent-gold"
              />
              <div className="grid gap-4 md:grid-cols-[1fr_0.9fr]">
                <svg viewBox="0 0 220 180" className="w-full">
                  <line x1="26" y1="150" x2="200" y2="150" stroke="rgba(255,255,255,0.08)" />
                  <line x1="26" y1="150" x2="26" y2="20" stroke="rgba(255,255,255,0.08)" />
                  {/* Base vector (your poem) */}
                  <line x1="26" y1="150" x2="186" y2="150" stroke="#C9A84C" strokeWidth="3" strokeLinecap="round" />
                  <text x="96" y="166" fill="#C9A84C" style={{ fontFamily: "DM Mono", fontSize: "9px" }}>your poem</text>
                  {/* Corpus vector */}
                  <line
                    x1="26" y1="150"
                    x2={26 + Math.cos((angle * Math.PI) / 180) * 140}
                    y2={150 - Math.sin((angle * Math.PI) / 180) * 140}
                    stroke={angle < 45 ? "#5CA88B" : angle < 90 ? "#C9A84C" : angle < 135 ? "#C97A5C" : "#8B1A1A"}
                    strokeWidth="3" strokeLinecap="round"
                  />
                  <text
                    x={26 + Math.cos((angle * Math.PI) / 180) * 75}
                    y={150 - Math.sin((angle * Math.PI) / 180) * 75 - 8}
                    fill="#6B6560" style={{ fontFamily: "DM Mono", fontSize: "9px" }}
                  >corpus line</text>
                  {/* Angle arc */}
                  <path
                    d={`M 66 150 A 40 40 0 0 1 ${26 + Math.cos((angle * Math.PI) / 180) * 40} ${150 - Math.sin((angle * Math.PI) / 180) * 40}`}
                    fill="none" stroke="rgba(255,255,255,0.25)" strokeWidth="1.5"
                  />
                  <text x="72" y="138" fill="#E8E0D0" style={{ fontFamily: "DM Mono", fontSize: "11px" }}>{angle}°</text>
                </svg>
                <div className="space-y-3">
                  <div className="metric-card rounded-[20px] p-4">
                    <p className="section-eyebrow text-[9px]">cos(θ)</p>
                    <p className="mt-2 font-display text-4xl" style={{
                      color: angle < 45 ? "#5CA88B" : angle < 90 ? "#C9A84C" : angle < 135 ? "#C97A5C" : "#8B1A1A"
                    }}>{cosineValue}</p>
                  </div>
                  <div className="rounded-2xl border border-white/5 bg-white/[0.02] p-3">
                    <p className="font-mono text-[9px] uppercase tracking-widest" style={{
                      color: angle < 45 ? "#5CA88B" : angle < 90 ? "#C9A84C" : angle < 135 ? "#C97A5C" : "#8B1A1A"
                    }}>
                      {angle < 30 ? "✦ Near-identical style" : angle < 60 ? "✓ Strong stylistic echo" : angle < 90 ? "~ Loosely related" : angle < 120 ? "✗ Divergent style" : "✗✗ Opposite register"}
                    </p>
                    <p className="mt-2 text-xs text-text-dim">
                      {angle < 30 ? "Poems share metre, vocabulary, and imagery patterns." : angle < 60 ? "Similar themes but distinct voice." : angle < 90 ? "Some shared words, different rhythm." : angle < 120 ? "Different poet, era, or form." : "Completely mismatched register."}
                    </p>
                  </div>
                </div>
              </div>
              <div className="flex gap-3 text-[9px] font-mono uppercase tracking-widest">
                {[["0°", "perfect", "#5CA88B"], ["45°", "similar", "#C9A84C"], ["90°", "unrelated", "#C97A5C"], ["135°+", "opposite", "#8B1A1A"]].map(([deg, label, color]) => (
                  <button key={deg} onClick={() => setAngle(parseInt(deg))} className="rounded-xl border border-white/5 bg-white/5 px-3 py-1.5 transition hover:bg-white/10" style={{ color }}>
                    {deg} {label}
                  </button>
                ))}
              </div>
            </div>
          </article>
        </section>

        <section data-reveal className="glass-panel rounded-[36px] p-8">
          <p className="section-eyebrow text-xs">Step 2</p>
          <h2 className="mt-3 font-display text-3xl text-text">Embedding space</h2>
          <p className="mt-4 max-w-3xl text-sm leading-7 text-text-dim">
            Each dot is a corpus excerpt. The white star is your input verse. Its nearest neighbors glow when style, imagery, and cadence cluster together.
          </p>
          <div className="mt-8">
            <EmbeddingPlot3D points={corpus} queryPoint={queryPoint} nearestNeighbors={sbert?.nearest_neighbors ?? []} />
          </div>
        </section>

        <section data-reveal className="glass-panel rounded-[36px] p-8">
          <div className="flex items-end justify-between gap-4">
            <div>
              <p className="section-eyebrow text-xs">Step 4</p>
              <h2 className="mt-3 font-display text-3xl text-text">Retrieved lines</h2>
            </div>
            {loadingExplain ? (
              <div className="w-48 overflow-hidden rounded-full bg-white/5">
                <motion.div
                  initial={{ x: "-100%" }}
                  animate={{ x: "100%" }}
                  transition={{ repeat: Number.POSITIVE_INFINITY, duration: 1.6, ease: "linear" }}
                  className="h-2 w-20 rounded-full bg-gradient-to-r from-transparent via-gold to-transparent"
                />
              </div>
            ) : null}
          </div>
          <div className="mt-8">
            <SimilarLines inputText={text} lines={retrievedLines} />
          </div>
        </section>

        <section data-reveal className="glass-panel rounded-[36px] p-8">
          <p className="section-eyebrow text-xs">Full architecture</p>
          <h2 className="mt-3 font-display text-3xl text-text">SBERT retrieval pipeline</h2>
          <div className="mt-8 overflow-x-auto">
            <SbertArchitecture />
          </div>
        </section>
      </div>
    </main>
  );
}
