import { Cpu, Database, Github, Orbit, ScrollText, Sparkles } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import AnimatedCounter from "../components/AnimatedCounter";
import useScrollReveal from "../hooks/useScrollReveal";
import { getDatasetStats } from "../lib/api";
import { POET_METADATA, PROJECT_META } from "../lib/poets";

const techBadges = [
  { label: "React + Vite", icon: Sparkles },
  { label: "FastAPI", icon: Cpu },
  { label: "Framer Motion", icon: Orbit },
  { label: "Three.js", icon: Database },
  { label: "DistilBERT", icon: ScrollText },
];

export default function About() {
  const containerRef = useRef(null);
  useScrollReveal(containerRef);

  const [stats, setStats] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    getDatasetStats()
      .then((response) => {
        if (active) setStats(response.stats);
      })
      .catch((err) => {
        if (active) setError(err.message);
      });
    return () => { active = false; };
  }, []);

  return (
    <main ref={containerRef} className="relative pb-28 pt-10 sm:pt-16">
      <div className="content-wrap space-y-14">
        {/* Header */}
        <section data-reveal className="grid gap-8 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="glass-panel rounded-[40px] p-10 sm:p-14">
            <p className="section-eyebrow text-xs">About Project</p>
            <h1 className="mt-5 font-display text-4xl uppercase tracking-[0.08em] text-text sm:text-6xl">
              A Neural-Linguistic Fusion Lab
            </h1>
            <p className="mt-6 max-w-3xl text-lg leading-8 text-text-dim">
              PoetryDNA is a research-grade application designed to identify poetic authorship by fusing deep-learning semantic understanding with classical stylometric feature engineering.
            </p>
          </div>
          <div className="grid gap-6">
            <div className="metric-card rounded-[32px] p-8">
              <p className="section-eyebrow text-[10px]">Notebook Accuracy</p>
              <p className="mt-4 font-display text-6xl text-gold">
                <AnimatedCounter value={PROJECT_META.accuracy} decimals={1} suffix="%" />
              </p>
            </div>
            <div className="metric-card rounded-[32px] p-8">
              <p className="section-eyebrow text-[10px]">Dual Models</p>
              <p className="mt-4 font-display text-6xl text-gold">
                <AnimatedCounter value={PROJECT_META.modelCount} />
              </p>
            </div>
          </div>
        </section>

        {/* Poets */}
        <section data-reveal className="glass-panel rounded-[36px] p-8">
          <div className="mb-8">
            <p className="section-eyebrow text-xs">The Corpus</p>
            <h2 className="mt-3 font-display text-3xl text-text">Six Literary Fingerprints</h2>
          </div>
          <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
            {Object.entries(POET_METADATA).map(([poet, meta]) => (
              <article key={poet} className="glass-panel rounded-[28px] p-6">
                <div className="h-1 w-16 rounded-full" style={{ background: meta.color }} />
                <h3 className="mt-5 font-display text-3xl text-text">{poet}</h3>
                <p className="mt-2 font-mono text-[10px] uppercase tracking-widest text-gold">{meta.era}</p>
                <div className="mt-5 flex flex-wrap gap-2">
                  {meta.styleTags.map((tag) => (
                    <span key={tag} className="rounded-full border border-white/5 bg-white/5 px-3 py-1 font-mono text-[10px] text-text-dim">
                      {tag}
                    </span>
                  ))}
                </div>
              </article>
            ))}
          </div>
        </section>

        {/* Tech */}
        <section data-reveal className="grid gap-6 lg:grid-cols-2">
          <article className="glass-panel rounded-[36px] p-8">
            <p className="section-eyebrow text-xs">Stack</p>
            <h2 className="mt-3 font-display text-3xl text-text">The Architecture</h2>
            <div className="mt-8 grid gap-4 sm:grid-cols-2">
              {techBadges.map((badge) => {
                const Icon = badge.icon;
                return (
                  <div key={badge.label} className="rounded-2xl border border-white/5 bg-white/5 p-5">
                    <Icon className="text-gold" size={24} />
                    <p className="mt-4 font-display text-xl text-text">{badge.label}</p>
                  </div>
                );
              })}
            </div>
          </article>

          <article className="glass-panel rounded-[36px] p-8">
            <p className="section-eyebrow text-xs">Dataset</p>
            <h2 className="mt-3 font-display text-3xl text-text">Corpus profile</h2>
            <div className="mt-8 grid gap-4 grid-cols-2">
              <div className="metric-card rounded-[24px] p-5">
                <p className="section-eyebrow text-[10px]">Samples</p>
                <p className="mt-3 font-display text-3xl text-gold">
                  <AnimatedCounter value={stats?.total_samples ?? PROJECT_META.trainingSamples} />
                </p>
              </div>
              <div className="metric-card rounded-[24px] p-5">
                <p className="section-eyebrow text-[10px]">Avg words</p>
                <p className="mt-3 font-display text-3xl text-gold">
                  <AnimatedCounter value={stats?.avg_word_count ?? 47} decimals={1} />
                </p>
              </div>
            </div>
            <div className="mt-8 overflow-hidden rounded-2xl border border-white/5 bg-black/20">
              <table className="w-full text-left">
                <thead className="bg-white/5">
                  <tr className="font-mono text-[10px] uppercase tracking-widest text-text-dim">
                    <th className="px-5 py-4">Poet</th>
                    <th className="px-5 py-4 text-right">Samples</th>
                  </tr>
                </thead>
                <tbody>
                  {(stats?.per_poet ?? []).map((entry) => (
                    <tr key={entry.poet} className="border-t border-white/5">
                      <td className="px-5 py-4 font-display text-lg text-text">{entry.poet}</td>
                      <td className="px-5 py-4 text-right font-mono text-xs text-gold">{entry.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>
        </section>

        {/* Team */}
        <section data-reveal className="glass-panel rounded-[36px] p-8">
          <div className="flex items-center justify-between mb-8">
            <div>
              <p className="section-eyebrow text-xs">The Lab</p>
              <h2 className="mt-3 font-display text-3xl text-text">Built by</h2>
            </div>
            <a
              href={PROJECT_META.githubUrl}
              target="_blank"
              rel="noreferrer"
              className="rounded-full border border-gold/20 bg-gold/5 p-4 text-gold hover:bg-gold/10 transition-colors"
            >
              <Github size={24} />
            </a>
          </div>
          <div className="grid gap-6 md:grid-cols-2">
            {PROJECT_META.team.map((member) => (
              <div key={member.name} className="rounded-[28px] border border-white/5 bg-white/5 p-8">
                <p className="font-display text-3xl text-text">{member.name}</p>
                <p className="mt-2 font-mono text-[10px] uppercase tracking-widest text-gold">{member.role}</p>
              </div>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
