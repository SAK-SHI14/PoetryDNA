import { motion } from "framer-motion";
import {
  BookOpen, Feather, Music, Type, AlignLeft, Hash,
  BarChart3, Zap, FileText
} from "lucide-react";
import { useDeferredValue, useEffect, useRef, useState, useTransition } from "react";
import ConfidenceBars from "../components/ConfidenceBars";
import useScrollReveal from "../hooks/useScrollReveal";
import { explainNlp } from "../lib/api";
import { SAMPLE_TEXT } from "../lib/poets";

const CATEGORY_META = {
  prosody: { icon: Music, label: "Prosody", color: "#C9A84C", desc: "Syllable counts, metrical variance, and iambic stress patterns." },
  rhyme: { icon: Feather, label: "Rhyme", color: "#8B1A1A", desc: "End-rhyme density and scheme detection (ABAB, AABB, ABBA)." },
  vocabulary: { icon: BookOpen, label: "Vocabulary", color: "#C9A84C", desc: "Archaic, nature, dark, divine, and sensory keyword ratios." },
  structure: { icon: AlignLeft, label: "Structure", color: "#8B1A1A", desc: "Line length, enjambment, repetition, and readability." },
  pos: { icon: Type, label: "Part-of-Speech", color: "#C9A84C", desc: "Noun, verb, adjective, adverb, and proper noun density." },
};

function FeatureBar({ name, value, maxValue }) {
  const barWidth = maxValue > 0 ? Math.min((Math.abs(value) / maxValue) * 100, 100) : 0;
  return (
    <div className="rounded-2xl border border-white/5 bg-black/20 p-4">
      <div className="flex items-center justify-between mb-2">
        <p className="font-mono text-[10px] uppercase tracking-widest text-gold">{name}</p>
        <p className="font-mono text-[10px] text-text-dim">{value.toFixed(4)}</p>
      </div>
      <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${barWidth}%` }}
          transition={{ duration: 0.7, ease: "easeOut" }}
          className="h-full rounded-full"
          style={{ background: value >= 0 ? "#C9A84C" : "#8B1A1A" }}
        />
      </div>
    </div>
  );
}

function CategorySection({ catKey, features }) {
  const meta = CATEGORY_META[catKey];
  if (!meta || !features?.length) return null;
  const Icon = meta.icon;
  // Show only the top 5 most discriminating features (by absolute value)
  const top5 = [...features]
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 5);
  const maxVal = Math.max(...top5.map((f) => Math.abs(f.value)), 0.001);

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="glass-panel rounded-[36px] p-6"
    >
      <div className="flex items-center gap-3 mb-2">
        <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-white/5 bg-white/5 shadow-gold-soft">
          <Icon size={24} style={{ color: meta.color }} />
        </div>
        <div>
          <h3 className="font-display text-2xl text-text">{meta.label}</h3>
          <p className="text-sm text-text-dim">{meta.desc}</p>
        </div>
      </div>
      <div className="mt-4 space-y-3">
        {top5.map((f) => (
          <FeatureBar key={f.name} name={f.name} value={f.value} maxValue={maxVal} />
        ))}
      </div>
      <p className="mt-3 font-mono text-[9px] uppercase tracking-widest text-text-dim">top 5 signals • sorted by strength</p>
    </motion.div>
  );
}

export default function LinguisticBrain() {
  const containerRef = useRef(null);
  useScrollReveal(containerRef);

  const [text, setText] = useState(SAMPLE_TEXT.nlp);
  const deferredText = useDeferredValue(text);
  const [payload, setPayload] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [isPending, startTransition] = useTransition();

  useEffect(() => {
    if (!deferredText.trim()) return undefined;

    let active = true;
    setLoading(true);
    setError("");

    const timer = window.setTimeout(async () => {
      try {
        const response = await explainNlp(deferredText);
        if (!active) return;
        startTransition(() => setPayload(response));
      } catch (err) {
        if (active) setError(err.message);
      } finally {
        if (active) setLoading(false);
      }
    }, 400);

    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [deferredText, startTransition]);

  const nlp = payload?.nlp;
  const categories = nlp?.categories ?? {};
  const tfidfTerms = nlp?.tfidf_top_terms ?? [];

  return (
    <main ref={containerRef} className="relative pb-28 pt-10 sm:pt-16">
      <div className="content-wrap space-y-14">
        {/* Header */}
        <section data-reveal className="relative overflow-hidden rounded-[40px] border border-gold/10 p-10 sm:p-14">
          <div className="absolute inset-0 hero-glow opacity-65" />
          <div className="relative z-10 max-w-4xl">
            <p className="section-eyebrow text-xs">Linguistic brain</p>
            <h1 className="mt-5 font-display text-5xl uppercase tracking-[0.08em] text-text sm:text-6xl lg:text-[78px]">
              Stylometric Fingerprint
            </h1>
            <p className="mt-6 max-w-3xl text-lg leading-8 text-text-dim">
              The model reads your poem through 5 lenses — prosody, rhyme, vocabulary, structure, and syntax — then asks: which poet leaves these exact fingerprints?
            </p>
          </div>
        </section>

        {/* Live prompt */}
        <section data-reveal className="glass-panel rounded-[36px] p-8">
          <div className="grid gap-8 lg:grid-cols-[1.1fr_0.9fr]">
            <div>
              <p className="section-eyebrow text-xs">Live analysis</p>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                className="mt-5 min-h-[220px] w-full rounded-[28px] border border-gold/15 bg-black/25 p-6 text-lg leading-8 text-text outline-none transition focus:border-gold/45"
              />
            </div>
            <div className="space-y-4">
              <div className="metric-card rounded-[28px] p-6">
                <p className="section-eyebrow text-[10px]">LightGBM prediction</p>
                <p className="mt-3 font-display text-4xl text-gold">
                  {nlp?.predicted_poet ?? "Listening..."}
                </p>
              </div>
              <div className="metric-card rounded-[28px] p-6">
                <p className="section-eyebrow text-[10px]">Confidence</p>
                <p className="mt-3 font-display text-5xl text-gold">
                  {nlp ? `${nlp.confidence}%` : "--"}
                </p>
              </div>
              <div className="metric-card rounded-[28px] p-6">
                <p className="section-eyebrow text-[10px]">State</p>
                <p className="mt-3 font-mono text-sm uppercase tracking-[0.28em] text-text-dim">
                  {loading ? "Extracting features" : isPending ? "Rendering" : "Ready"}
                </p>
              </div>
            </div>
          </div>
          {error && (
            <p className="mt-6 rounded-2xl border border-crimson/20 bg-crimson/10 px-4 py-3 text-sm text-[#d7a3a3]">
              {error}
            </p>
          )}
        </section>

        {/* Probability ladder */}
        {nlp?.top_poets && (
          <section data-reveal className="glass-panel rounded-[36px] p-8">
            <p className="section-eyebrow text-xs">Probability ladder</p>
            <h3 className="mt-3 font-display text-3xl text-text">LightGBM ranking</h3>
            <div className="mt-8">
              <ConfidenceBars items={nlp.top_poets} />
            </div>
          </section>
        )}

        {/* Feature highlights */}
        {nlp?.feature_highlights?.length > 0 && (
          <section data-reveal className="glass-panel rounded-[36px] p-8">
            <div className="mb-6">
              <p className="section-eyebrow text-xs">Top discriminating signals</p>
              <h3 className="mt-3 font-display text-3xl text-text">Feature highlights</h3>
            </div>
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
              {nlp.feature_highlights.map((f) => (
                <div key={f.name} className="rounded-[24px] border border-gold/10 bg-white/5 p-5">
                  <div className="flex items-center gap-2 mb-2">
                    <Zap size={14} className="text-gold" />
                    <span className="font-mono text-[10px] uppercase tracking-widest text-gold">{f.name}</span>
                  </div>
                  <p className="font-display text-3xl text-text">{f.value.toFixed(4)}</p>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Feature Categories */}
        {Object.keys(categories).length > 0 && (
          <section data-reveal className="space-y-6">
            <div>
              <p className="section-eyebrow text-xs">Hand-crafted features</p>
              <h2 className="mt-3 font-display text-3xl text-text">Top 5 signals per category</h2>
              <p className="mt-3 max-w-2xl text-sm leading-7 text-text-dim">
                Only the five most decisive signals from each category are shown. The bar width reflects how strongly that feature pushes the classifier toward its prediction.
              </p>
            </div>
            <div className="grid gap-6 md:grid-cols-2">
              {Object.entries(categories).map(([key, features]) => (
                <CategorySection key={key} catKey={key} features={features} />
              ))}
            </div>
          </section>
        )}

        {/* TF-IDF Top Terms */}
        {tfidfTerms.length > 0 && (
          <section data-reveal className="glass-panel rounded-[36px] p-8">
            <div className="flex items-center gap-4 mb-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-white/5 bg-white/5">
                <FileText size={28} className="text-gold" />
              </div>
              <div>
                <p className="section-eyebrow text-xs">TF-IDF Analysis</p>
                <h3 className="mt-1 font-display text-2xl text-text">Words that define this verse</h3>
              </div>
            </div>
            <div className="mb-6 rounded-2xl border border-gold/10 bg-gold/5 p-4">
              <p className="text-sm leading-7 text-text-dim">
                <span className="text-gold font-mono">TF-IDF</span> (Term Frequency – Inverse Document Frequency) finds words that appear often in <em>this</em> poem but rarely in the overall corpus. A high score means the word is a distinctive stylistic marker, not just common filler. These are the terms that made the classifier lean toward its verdict.
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
              {tfidfTerms.slice(0, 9).map((t, i) => {
                const maxScore = tfidfTerms[0]?.score ?? 1;
                const width = Math.round((t.score / maxScore) * 100);
                return (
                  <div key={t.term} className="rounded-2xl border border-white/5 bg-black/20 px-5 py-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-display text-lg text-text">{t.term}</span>
                      <span className="font-mono text-[10px] text-gold">{t.score.toFixed(3)}</span>
                    </div>
                    <div className="h-1 rounded-full bg-white/5 overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${width}%` }}
                        transition={{ duration: 0.7, delay: i * 0.06 }}
                        className="h-full rounded-full bg-gold"
                      />
                    </div>
                    <p className="mt-1 font-mono text-[9px] text-text-dim">rank #{i + 1} distinctive term</p>
                  </div>
                );
              })}
            </div>
          </section>
        )}

        {/* Model summary */}
        <section data-reveal className="glass-panel rounded-[36px] p-8">
          <div className="flex items-center gap-4 mb-6">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-white/5 bg-white/5">
              <BarChart3 size={28} className="text-gold" />
            </div>
            <div>
              <p className="section-eyebrow text-xs">Architecture</p>
              <h3 className="mt-1 font-display text-2xl text-text">LightGBM ensemble</h3>
            </div>
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            {[
              ["Algorithm", "Gradient Boosted Trees"],
              ["Total Features", "232 Dimensions"],
              ["Standalone Accuracy", "81.36%"],
              ["Training Samples", "1,996 stanzas"],
              ["Feature Groups", "Prosody · Rhyme · Vocab · Structure · POS"],
              ["Representations", "TF-IDF (SVD-150) + Char n-gram (SVD-50)"],
            ].map(([key, val]) => (
              <div key={key} className="flex justify-between border-b border-white/5 pb-2">
                <span className="text-sm text-text-dim">{key}</span>
                <span className="font-mono text-sm text-gold">{val}</span>
              </div>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
