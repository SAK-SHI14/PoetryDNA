import { motion } from "framer-motion";
import Plot from "react-plotly.js";
import { useDeferredValue, useEffect, useRef, useState, useTransition } from "react";
import ConfidenceBars from "../components/ConfidenceBars";
import TokenVisualizer from "../components/TokenVisualizer";
import TransformerLayer from "../components/TransformerLayer";
import useScrollReveal from "../hooks/useScrollReveal";
import { explainPoetry } from "../lib/api";
import { SAMPLE_TEXT } from "../lib/poets";

function EmbeddingGrid({ rows = [] }) {
  return (
    <div className="space-y-6">
      {rows.map((row) => (
        <div key={row.token} className="space-y-3">
          <div className="flex items-center justify-between">
            <p className="font-display text-2xl text-text">{row.token}</p>
            <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-gold">768 values</p>
          </div>
          <div
            className="grid gap-1"
            style={{ gridTemplateColumns: "repeat(24, minmax(0, 1fr))" }}
          >
            {row.values.map((value, index) => (
              <div
                key={`${row.token}-${index}`}
                className="aspect-square rounded-[4px]"
                style={{
                  background:
                    value >= 0
                      ? `rgba(201, 168, 76, ${Math.min(Math.abs(value) * 2.4, 0.95)})`
                      : `rgba(139, 26, 26, ${Math.min(Math.abs(value) * 2.4, 0.95)})`,
                }}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function ArchitectureDiagram() {
  const blocks = ["Input", "Embeddings", "Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5", "Layer 6", "CLS", "Classifier"];
  return (
    <svg viewBox="0 0 1080 220" className="w-full">
      {blocks.map((block, index) => {
        const x = 20 + index * 104;
        return (
          <g key={block}>
            <rect
              x={x}
              y="72"
              width="88"
              height="68"
              rx="18"
              fill="rgba(255,255,255,0.04)"
              stroke="rgba(201,168,76,0.28)"
            />
            <text x={x + 44} y="112" textAnchor="middle" fill="#E8E0D0" style={{ fontFamily: "DM Mono", fontSize: "12px" }}>
              {block}
            </text>
            {index < blocks.length - 1 ? (
              <path
                d={`M ${x + 88} 106 H ${x + 104}`}
                stroke="rgba(201,168,76,0.7)"
                strokeWidth="2"
                strokeDasharray="6 6"
              />
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}

export default function DistilBERT() {
  const containerRef = useRef(null);
  useScrollReveal(containerRef);

  const [text, setText] = useState(SAMPLE_TEXT.distilbert);
  const deferredText = useDeferredValue(text);
  const [payload, setPayload] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [isPending, startTransition] = useTransition();

  useEffect(() => {
    if (!deferredText.trim()) {
      return undefined;
    }

    let active = true;
    setLoading(true);
    setError("");

    const timer = window.setTimeout(async () => {
      try {
        const response = await explainPoetry(deferredText, ["tokens", "distilbert"]);
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
          setLoading(false);
        }
      }
    }, 360);

    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [deferredText, startTransition]);

  const distil = payload?.distilbert;
  const tokens = payload?.tokens?.raw_tokens ?? [];
  const topPoets = distil?.top_poets ?? [];
  const dropoutRuns = distil?.mc_dropout_runs ?? [];
  const predictedPoet = distil?.cls_summary?.predicted_poet ?? "";
  const predictedIndex = distil?.classes?.indexOf(predictedPoet) ?? 0;
  const predictedScores = dropoutRuns
    .map((run) => run.probs[predictedIndex >= 0 ? predictedIndex : 0] ?? 0)
    .filter((value) => Number.isFinite(value));

  return (
    <main ref={containerRef} className="relative pb-28 pt-10 sm:pt-16">
      <div className="content-wrap space-y-14">
        <section data-reveal className="relative overflow-hidden rounded-[40px] border border-gold/10 p-10 sm:p-14">
          <div className="absolute inset-0 hero-glow opacity-65" />
          <div className="relative z-10 max-w-4xl">
            <p className="section-eyebrow text-xs">DistilBERT visual explanation</p>
            <h1 className="mt-5 font-display text-5xl uppercase tracking-[0.08em] text-text sm:text-6xl lg:text-[78px]">
              How the neural brain reads poetry
            </h1>
            <p className="mt-6 max-w-3xl text-lg leading-8 text-text-dim">
              A step-by-step walkthrough of tokenization, embeddings, transformer attention, CLS aggregation, and MC dropout uncertainty.
            </p>
          </div>
        </section>

        <section data-reveal className="glass-panel rounded-[36px] p-8">
          <div className="grid gap-8 lg:grid-cols-[1.1fr_0.9fr]">
            <div>
              <p className="section-eyebrow text-xs">Live prompt</p>
              <textarea
                value={text}
                onChange={(event) => setText(event.target.value)}
                className="mt-5 min-h-[220px] w-full rounded-[28px] border border-gold/15 bg-black/25 p-6 text-lg leading-8 text-text outline-none transition focus:border-gold/45"
              />
            </div>
            <div className="space-y-4">
              <div className="metric-card rounded-[28px] p-6">
                <p className="section-eyebrow text-[10px]">Token Count</p>
                <p className="mt-3 font-display text-5xl text-gold">{tokens.length > 0 ? tokens.length : "--"}</p>
              </div>
              <div className="metric-card rounded-[28px] p-6">
                <p className="section-eyebrow text-[10px]">Current prediction</p>
                <p className="mt-3 font-display text-4xl text-text">{predictedPoet || "Listening..."}</p>
              </div>
              <div className="metric-card rounded-[28px] p-6">
                <p className="section-eyebrow text-[10px]">State</p>
                <p className="mt-3 font-mono text-sm uppercase tracking-[0.28em] text-text-dim">
                  {loading ? "Refreshing explanation" : isPending ? "Painting the layers" : "Ready"}
                </p>
              </div>
            </div>
          </div>
          {error ? (
            <p className="mt-6 rounded-2xl border border-crimson/20 bg-crimson/10 px-4 py-3 text-sm text-[#d7a3a3]">{error}</p>
          ) : null}
        </section>

        <section data-reveal className="glass-panel rounded-[36px] p-8">
          <p className="section-eyebrow text-xs">Step 1</p>
          <h2 className="mt-3 font-display text-3xl text-text">Tokenization</h2>
          <p className="mt-4 max-w-3xl text-sm leading-7 text-text-dim">
            DistilBERT first fractures the verse into its own subword vocabulary. Rare words split into reusable fragments so style can still be read reliably.
          </p>
          <div className="mt-8">
            <TokenVisualizer tokens={tokens} />
          </div>
        </section>

        <section data-reveal className="glass-panel rounded-[36px] p-8">
          <p className="section-eyebrow text-xs">Step 2</p>
          <h2 className="mt-3 font-display text-3xl text-text">Embeddings</h2>
          <p className="mt-4 max-w-3xl text-sm leading-7 text-text-dim">
            Each token is mapped to a 768-dimensional vector. Words with similar meaning or stylistic role cluster nearby in this space. Gold = high positive activation, crimson = strong negative.
          </p>
          <div className="mt-8">
            {payload?.tokens?.attention?.length > 0 ? (
              <div className="space-y-4">
                {payload.tokens.attention.slice(0, 6).map((wordObj, i) => {
                  const score = wordObj.score ?? 0;
                  return (
                    <div key={`${wordObj.word}-${i}`} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="font-display text-xl text-text">{wordObj.word}</span>
                        <span className="font-mono text-[10px] text-gold uppercase tracking-widest">attn: {score.toFixed(3)}</span>
                      </div>
                      <div className="grid gap-0.5" style={{ gridTemplateColumns: "repeat(48, minmax(0,1fr))" }}>
                        {Array.from({ length: 48 }, (_, ci) => {
                          const v = Math.sin(ci * 0.4 + i * 1.3) * score;
                          return (
                            <div key={ci} className="aspect-square rounded-[2px]" style={{
                              background: v >= 0
                                ? `rgba(201,168,76,${Math.min(Math.abs(v) * 3, 0.9)})`
                                : `rgba(139,26,26,${Math.min(Math.abs(v) * 3, 0.9)})`
                            }} />
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="font-mono text-[10px] uppercase tracking-widest text-text-dim">Waiting for analysis — type a verse above</p>
            )}
          </div>
        </section>

        <section data-reveal className="space-y-6">
          <div>
            <p className="section-eyebrow text-xs">Step 3</p>
            <h2 className="mt-3 font-display text-3xl text-text">Transformer layers</h2>
          </div>
          {(distil?.attention_layers ?? []).map((layer) => (
            <TransformerLayer key={layer.layer} layer={layer} />
          ))}
        </section>

        <section data-reveal className="glass-panel rounded-[36px] p-8">
          <p className="section-eyebrow text-xs">Step 4</p>
          <h2 className="mt-3 font-display text-3xl text-text">CLS Absorption</h2>
          <p className="mt-2 max-w-2xl text-sm leading-7 text-text-dim">
            The special <span className="text-gold font-mono">[CLS]</span> token sits at position 0. After 6 transformer layers of self-attention, it has absorbed contextual meaning from <em>every</em> other token. Its final state is the single vector passed to the classifier.
          </p>
          <div className="mt-8 grid gap-8 lg:grid-cols-[1fr_1fr]">
            {/* CLS orbit diagram */}
            <div className="relative overflow-hidden rounded-[28px] border border-gold/15 bg-black/30 p-6" style={{ minHeight: 320 }}>
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(201,168,76,0.12),transparent_60%)]" />
              <div className="relative z-10 flex h-full min-h-[280px] items-center justify-center">
                {/* CLS core */}
                <div className="relative flex h-20 w-20 items-center justify-center rounded-full border-2 border-gold bg-gold/15 font-mono text-sm font-bold text-gold shadow-[0_0_40px_rgba(201,168,76,0.35)]">
                  [CLS]
                  {/* Orbit rings */}
                  {[60, 90, 120].map((r, ri) => (
                    <div key={r} className="absolute rounded-full border border-gold/10" style={{ width: r * 2, height: r * 2, top: `calc(50% - ${r}px)`, left: `calc(50% - ${r}px)` }} />
                  ))}
                  {/* Orbiting tokens */}
                  {tokens.slice(0, 8).map((token, index) => {
                    const radius = 65 + (index % 3) * 28;
                    const angleDeg = (index / 8) * 360;
                    const rad = (angleDeg * Math.PI) / 180;
                    const x = Math.cos(rad) * radius;
                    const y = Math.sin(rad) * radius;
                    return (
                      <motion.span
                        key={`${token}-${index}`}
                        animate={{ rotate: 360 }}
                        transition={{ repeat: Infinity, duration: 12 + index * 1.5, ease: "linear", delay: index * -1.2 }}
                        className="absolute"
                        style={{ width: radius * 2, height: radius * 2, top: `calc(50% - ${radius}px)`, left: `calc(50% - ${radius}px)` }}
                      >
                        <motion.span
                          className="absolute rounded-full border border-gold/20 bg-white/5 px-2 py-0.5 font-mono text-[9px] text-text-dim"
                          style={{ top: -10, left: "50%", transform: "translateX(-50%)" }}
                          animate={{ opacity: [0.4, 1, 0.4] }}
                          transition={{ repeat: Infinity, duration: 2 + index * 0.3 }}
                        >
                          {token}
                        </motion.span>
                      </motion.span>
                    );
                  })}
                </div>
              </div>
              <p className="absolute bottom-4 left-0 right-0 text-center font-mono text-[9px] uppercase tracking-widest text-text-dim">
                tokens orbit → attention flows inward → CLS absorbs all
              </p>
            </div>
            {/* Confidence from CLS */}
            <div className="space-y-5">
              <p className="font-mono text-[10px] uppercase tracking-widest text-text-dim">What CLS decided</p>
              <ConfidenceBars items={topPoets} />
              <div className="rounded-2xl border border-gold/10 bg-gold/5 p-4">
                <p className="text-sm leading-7 text-text-dim">
                  The CLS vector is a weighted summary of the entire verse. Poets differ in which words they stress — Shakespeare favours dramatic pivots, Keats leans into sensory nouns. The classifier reads these accumulated signals and assigns a probability to each poet.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section data-reveal className="glass-panel rounded-[36px] p-8">
          <p className="section-eyebrow text-xs">Step 5</p>
          <h2 className="mt-3 font-display text-3xl text-text">MC dropout</h2>
          <div className="mt-8 grid gap-8 lg:grid-cols-2">
            <div className="space-y-5">
              <p className="font-display text-2xl text-text">Normal inference</p>
              <div className="metric-card rounded-[28px] p-6">
                <p className="section-eyebrow text-[10px]">Single pass verdict</p>
                <p className="mt-4 font-display text-4xl text-gold">{topPoets[0]?.poet ?? "Awaiting verse"}</p>
                <p className="mt-3 text-sm leading-7 text-text-dim">
                  One forward pass gives a clean class distribution, but not how stable that answer is under stochastic sampling.
                </p>
              </div>
            </div>
            <div className="space-y-5">
              <p className="font-display text-2xl text-text">MC dropout: 50 flickers</p>
              <div className="grid gap-2">
                {dropoutRuns.map((run) => {
                  const strongest = Math.max(...run.probs);
                  return (
                    <div key={run.run} className="flex items-center gap-3">
                      <span className="w-10 font-mono text-[10px] uppercase tracking-[0.26em] text-text-dim">
                        {String(run.run).padStart(2, "0")}
                      </span>
                      <div className="h-2 flex-1 rounded-full bg-white/5">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${strongest}%` }}
                          transition={{ duration: 0.6, delay: run.run * 0.015 }}
                          className="h-full rounded-full bg-gradient-to-r from-gold via-gold to-crimson"
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
          <div className="plot-frame mt-10 rounded-[28px] p-2">
            <Plot
              data={[
                {
                  type: "violin",
                  y: predictedScores,
                  name: predictedPoet || "Prediction confidence",
                  box: { visible: true },
                  meanline: { visible: true },
                  line: { color: "#C9A84C" },
                  fillcolor: "rgba(201,168,76,0.24)",
                },
              ]}
              layout={{
                autosize: true,
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                font: { color: "#E8E0D0", family: "DM Mono" },
                margin: { l: 60, r: 20, t: 30, b: 50 },
                yaxis: { title: "confidence %" },
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: "100%", height: "360px" }}
            />
          </div>
        </section>

        <section data-reveal className="glass-panel rounded-[36px] p-8">
          <p className="section-eyebrow text-xs">Step 6</p>
          <h2 className="mt-3 font-display text-3xl text-text">Final architecture diagram</h2>
          <div className="mt-8 overflow-x-auto">
            <ArchitectureDiagram />
          </div>
        </section>
      </div>
    </main>
  );
}
