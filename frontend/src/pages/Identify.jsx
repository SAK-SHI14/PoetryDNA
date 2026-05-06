import { AnimatePresence, motion } from "framer-motion";
import { Feather, Leaf, MoonStar, Music, Brain, Cpu, GitMerge } from "lucide-react";
import { useRef, useState, useTransition } from "react";
import AttentionHeatmap from "../components/AttentionHeatmap";
import ConfidenceBars from "../components/ConfidenceBars";
import DNAHelix3D from "../components/DNAHelix3D";
import SimilarLines from "../components/SimilarLines";
import useScrollReveal from "../hooks/useScrollReveal";
import useTypewriter from "../hooks/useTypewriter";
import { explainPoetry, predictPoet } from "../lib/api";
import { POET_METADATA, SAMPLE_TEXT } from "../lib/poets";

const styleIcons = [MoonStar, Music, Leaf, Feather];

function HelixMeter({ confidence, uncertainty }) {
  return (
    <div className="grid gap-6 md:grid-cols-[1.3fr_0.7fr] md:items-center">
      <div className="relative overflow-hidden rounded-[32px] border border-white/5 bg-black/20">
        <DNAHelix3D confidence={confidence} className="relative z-10" />
        <p className="absolute bottom-3 left-0 right-0 text-center font-mono text-[10px] uppercase tracking-widest text-text-dim/50">
          Drag to Rotate
        </p>
      </div>
      <div className="space-y-4">
        <div className="metric-card rounded-[28px] p-6">
          <p className="section-eyebrow text-[10px]">Confidence</p>
          <p className="mt-3 font-display text-5xl text-gold">{confidence.toFixed(1)}%</p>
        </div>
        <div className="metric-card rounded-[28px] p-6">
          <p className="section-eyebrow text-[10px]">Uncertainty</p>
          <p className="mt-3 font-mono text-2xl text-text">{uncertainty.toFixed(4)}</p>
        </div>
      </div>
    </div>
  );
}

function ResultName({ name }) {
  return (
    <h2 className="flex flex-wrap gap-x-2 font-display text-4xl uppercase tracking-[0.08em] text-text sm:text-6xl lg:text-[74px]">
      {name.split("").map((letter, index) => (
        <motion.span
          key={`${letter}-${index}`}
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.03, duration: 0.5 }}
        >
          {letter === " " ? "\u00A0" : letter}
        </motion.span>
      ))}
    </h2>
  );
}

function LoadingResults() {
  return (
    <div className="glass-panel rounded-[36px] p-8">
      <div className="ink-loader flex min-h-[260px] items-center justify-center">
        <div className="h-28 w-28 animate-pulse rounded-full border border-gold/20 shadow-gold-glow" />
      </div>
      <div className="mt-6 space-y-4">
        <div className="h-5 w-1/3 rounded-full bg-white/5" />
        <div className="h-12 w-2/3 rounded-full bg-white/5" />
        <div className="h-32 w-full rounded-3xl bg-white/5" />
      </div>
    </div>
  );
}

export default function Identify() {
  const containerRef = useRef(null);
  useScrollReveal(containerRef);

  const [text, setText] = useState(SAMPLE_TEXT.identify);
  const [prediction, setPrediction] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [isPending, startTransition] = useTransition();

  const typedHeading = useTypewriter("WHO WROTE THIS?");

  async function handleSubmit(event) {
    event.preventDefault();
    if (!text.trim()) {
      setError("Paste a verse to awaken the archive.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const [predictPayload, explainPayload] = await Promise.all([
        predictPoet(text),
        explainPoetry(text, ["distilbert"]),
      ]);

      startTransition(() => {
        setPrediction(predictPayload);
        setExplanation(explainPayload);
      });
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  const poetMeta = prediction ? POET_METADATA[prediction.predicted_poet] : null;

  return (
    <main ref={containerRef} className="relative pb-28 pt-10 sm:pt-16">
      <div className="content-wrap space-y-14">
        {/* ── Hero ── */}
        <section
          data-reveal
          className="relative overflow-hidden rounded-[40px] border border-gold/10 p-10 sm:p-14"
        >
          <div className="absolute inset-0 hero-glow opacity-65" />
          <div className="relative z-10 mx-auto max-w-4xl text-center">
            <p className="section-eyebrow text-xs">PoetryDNA Attribution</p>
            <h1 className="mt-6 font-display text-4xl uppercase tracking-[0.1em] text-text sm:text-6xl lg:text-[78px]">
              {typedHeading}
              <span className="inline-block h-[0.9em] w-[2px] animate-pulse bg-gold align-middle" />
            </h1>
            <p className="mx-auto mt-6 max-w-2xl text-lg leading-8 text-text-dim">
              Paste any verse. Our Neural-Linguistic Fusion engine will scan the patterns to identify the author.
            </p>
          </div>
        </section>

        {/* ── Input + Info ── */}
        <section data-reveal className="grid gap-8 lg:grid-cols-[1.1fr_0.9fr]">
          <form onSubmit={handleSubmit} className="glass-panel rounded-[36px] p-6 sm:p-8">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="section-eyebrow text-xs">Verse Intake</p>
                <h2 className="mt-2 font-display text-3xl text-text">Identify the Poet</h2>
              </div>
              <p className="font-mono text-[10px] uppercase tracking-widest text-text-dim">
                Max 128 Tokens
              </p>
            </div>
            <textarea
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="From fairest creatures we desire increase..."
              className="mt-6 min-h-[360px] w-full resize-none rounded-[28px] border border-gold/15 bg-black/25 p-6 font-body text-lg leading-8 text-text outline-none transition focus:border-gold/45"
            />
            <div className="mt-6 flex flex-wrap items-center gap-4">
              <button type="submit" disabled={loading} className="ink-button">
                {loading ? "Analysing" : "Analyse"}
              </button>
              <p className="font-mono text-[10px] uppercase tracking-widest text-text-dim">
                neural + linguistic brain fusion
              </p>
            </div>
            {error ? (
              <p className="mt-4 rounded-2xl border border-crimson/20 bg-crimson/10 px-4 py-3 text-sm text-[#d7a3a3]">
                {error}
              </p>
            ) : null}
          </form>

          <div className="space-y-6">
            <div className="glass-panel rounded-[36px] p-8">
              <p className="section-eyebrow text-xs">Inference ritual</p>
              <h2 className="mt-4 font-display text-3xl text-text">What the model returns</h2>
              <ul className="mt-6 space-y-4 text-lg leading-8 text-text-dim">
                <li>Predicted poet and calibrated confidence.</li>
                <li>Uncertainty score revealing stability.</li>
                <li>Nearest corpus parallels from the predicted poet.</li>
                <li>Attention heatmap showing stylistic anchors.</li>
              </ul>
            </div>
            <div className="grid gap-6 sm:grid-cols-2">
              <div className="metric-card rounded-[28px] p-6">
                <p className="section-eyebrow text-[10px]">Dual brain</p>
                <p className="mt-4 font-display text-2xl text-gold">Fusion model</p>
                <p className="mt-3 text-sm leading-7 text-text-dim">
                  Fusing DistilBERT's deep semantics with LightGBM's stylometric fingerprints.
                </p>
              </div>
              <div className="metric-card rounded-[28px] p-6">
                <p className="section-eyebrow text-[10px]">Explainable AI</p>
                <p className="mt-4 font-display text-2xl text-gold">Interpretable</p>
                <p className="mt-3 text-sm leading-7 text-text-dim">
                  Visualizing attention and linguistic features for model transparency.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* ── Results ── */}
        <AnimatePresence mode="wait">
          {loading ? (
            <motion.section
              key="loading"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -16 }}
            >
              <LoadingResults />
            </motion.section>
          ) : prediction ? (
            <motion.section
              key="results"
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              className="space-y-8"
            >
              {/* Primary attribution */}
              <section data-reveal className="glass-panel rounded-[40px] p-8 sm:p-10">
                <div className="flex flex-wrap items-start justify-between gap-6">
                  <div>
                    <p className="section-eyebrow text-xs">Primary attribution</p>
                    <ResultName name={prediction.predicted_poet} />
                    <span className="mt-4 inline-flex rounded-full border border-gold/20 bg-gold/5 px-5 py-2 font-mono text-[10px] uppercase tracking-widest text-gold">
                      {poetMeta?.era}
                    </span>
                  </div>
                  <div className="rounded-[28px] border border-white/5 bg-white/[0.03] px-5 py-4 text-right">
                    <p className="font-mono text-[10px] uppercase tracking-widest text-text-dim">signal strength</p>
                    <p className="mt-2 font-display text-4xl text-gold">{prediction.confidence.toFixed(1)}%</p>
                  </div>
                </div>
                <div className="mt-10">
                  <HelixMeter confidence={prediction.confidence} uncertainty={prediction.uncertainty} />
                </div>
              </section>

              {/* Style tags */}
              <section data-reveal className="space-y-6">
                <div className="flex items-end justify-between gap-4">
                  <div>
                    <p className="section-eyebrow text-xs">Style analysis</p>
                    <h3 className="mt-3 font-display text-3xl text-text">The toneprint</h3>
                  </div>
                </div>
                <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-4">
                  {poetMeta?.styleTags.map((tag, index) => {
                    const Icon = styleIcons[index] ?? Feather;
                    return (
                      <motion.article
                        key={tag}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.08 * index }}
                        className="glass-panel rounded-[28px] p-6"
                      >
                        <Icon className="text-gold" size={28} />
                        <h4 className="mt-5 font-display text-2xl text-text">{tag}</h4>
                        <p className="mt-3 text-sm leading-7 text-text-dim">
                          Detected recurring stylistic fingerprints in the verse.
                        </p>
                      </motion.article>
                    );
                  })}
                </div>
              </section>

              {/* Probability ladder */}
              <section data-reveal className="glass-panel rounded-[36px] p-8">
                <p className="section-eyebrow text-xs">Runner-up Poets</p>
                <h3 className="mt-3 font-display text-3xl text-text">Probability Ladder</h3>
                <div className="mt-8">
                  {(prediction.fusion_breakdown?.dl_top_poets?.length > 0) ? (
                    <ConfidenceBars items={prediction.fusion_breakdown.dl_top_poets} />
                  ) : (
                    <p className="font-mono text-[10px] uppercase tracking-widest text-text-dim">
                      Neural Brain offline — DL probabilities unavailable
                    </p>
                  )}
                </div>
              </section>

              {/* Fusion Breakdown */}
              {prediction.fusion_breakdown && (
                <section data-reveal className="glass-panel rounded-[36px] p-8">
                  <div className="mb-6">
                    <p className="section-eyebrow text-xs">Dual-brain fusion</p>
                    <h3 className="mt-3 font-display text-3xl text-text">Brain alignment</h3>
                  </div>
                  <div className="grid gap-6 md:grid-cols-2">
                    {/* Neural Brain */}
                    <div className="rounded-[28px] border border-gold/10 bg-white/5 p-5">
                      <div className="flex items-center gap-3 mb-3">
                        <Brain size={20} className="text-gold" />
                        <span className="font-mono text-[10px] uppercase tracking-widest text-gold">Neural Brain</span>
                        <span className="ml-auto font-mono text-[10px] text-text-dim">
                          {prediction.fusion_breakdown.dl_weight * 100}% weight
                        </span>
                      </div>
                      {prediction.fusion_breakdown.dl_prediction === "Unknown" ? (
                        <p className="font-mono text-[10px] uppercase tracking-widest text-text-dim mt-2">
                          ⚠ DL Engine offline — awaiting HF Space
                        </p>
                      ) : (
                        <>
                          <p className="font-display text-2xl text-text">
                            {prediction.fusion_breakdown.dl_prediction}
                          </p>
                          <div className="mt-3 h-1.5 rounded-full bg-white/5 overflow-hidden">
                            <div
                              className="h-full rounded-full bg-gold transition-all duration-700"
                              style={{ width: `${prediction.fusion_breakdown.dl_confidence}%` }}
                            />
                          </div>
                          <p className="mt-1 font-mono text-[10px] text-text-dim">
                            {prediction.fusion_breakdown.dl_confidence}% confidence
                          </p>
                        </>
                      )}
                    </div>
                    {/* Linguistic Brain */}
                    <div className="rounded-[28px] border border-white/10 bg-white/5 p-5">
                      <div className="flex items-center gap-3 mb-3">
                        <Cpu size={20} className="text-text-dim" />
                        <span className="font-mono text-[10px] uppercase tracking-widest text-text-dim">Linguistic Brain</span>
                        <span className="ml-auto font-mono text-[10px] text-text-dim">
                          {prediction.fusion_breakdown.nlp_weight * 100}% weight
                        </span>
                      </div>
                      <p className="font-display text-2xl text-text">
                        {prediction.fusion_breakdown.nlp_prediction}
                      </p>
                      <div className="mt-3 h-1.5 rounded-full bg-white/5 overflow-hidden">
                        <div
                          className="h-full rounded-full bg-text-dim transition-all duration-700"
                          style={{ width: `${prediction.fusion_breakdown.nlp_confidence}%` }}
                        />
                      </div>
                      <p className="mt-1 font-mono text-[10px] text-text-dim">
                        {prediction.fusion_breakdown.nlp_confidence}% confidence
                      </p>
                    </div>
                  </div>
                  {/* Fused Verdict */}
                  <div className="mt-6 rounded-[24px] border border-gold/20 bg-gold/5 p-5">
                    <div className="flex items-center gap-3 mb-2">
                      <GitMerge size={20} className="text-gold" />
                      <span className="font-mono text-[10px] uppercase tracking-widest text-gold">Fused Verdict</span>
                    </div>
                    <div className="flex items-end gap-4">
                      <p className="font-display text-4xl text-text">
                        {prediction.fusion_breakdown.fusion_prediction}
                      </p>
                      <p className="mb-1 font-display text-3xl text-gold">
                        {prediction.fusion_breakdown.fusion_confidence}%
                      </p>
                    </div>
                  </div>
                </section>
              )}

              {/* Similar lines */}
              <section data-reveal className="space-y-6">
                <div>
                  <p className="section-eyebrow text-xs">Similarity brain</p>
                  <h3 className="mt-3 font-display text-3xl text-text">Kindred lines in the corpus</h3>
                </div>
                <SimilarLines inputText={text} lines={prediction.similar_lines} />
              </section>

              {/* Attention heatmap */}
              <section data-reveal className="glass-panel rounded-[36px] p-8">
                <div className="mb-8">
                  <p className="section-eyebrow text-xs">Attention Heatmap</p>
                  <h3 className="mt-3 font-display text-3xl text-text">Words That Gave It Away</h3>
                </div>
                {prediction.attention?.length > 0 ? (
                  <AttentionHeatmap attention={prediction.attention} />
                ) : (
                  <div className="flex items-center gap-3 rounded-2xl border border-white/5 bg-white/[0.02] p-6">
                    <span className="h-2 w-2 animate-pulse rounded-full bg-gold" />
                    <p className="font-mono text-[10px] uppercase tracking-widest text-text-dim">
                      Neural Brain connecting — attention data will appear once HF Space is live
                    </p>
                  </div>
                )}
              </section>
            </motion.section>
          ) : null}
        </AnimatePresence>

        {isPending ? (
          <p className="text-center font-mono text-[10px] uppercase tracking-widest text-gold">
            recalibrating the neural manifold...
          </p>
        ) : null}
      </div>
    </main>
  );
}
