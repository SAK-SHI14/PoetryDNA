import { motion } from "framer-motion";

function normalizeWord(word) {
  return word.toLowerCase().replace(/[^a-z']/g, "");
}

function renderHighlightedLine(text, highlights) {
  return text.split(/(\s+)/).map((chunk, index) => {
    const normalized = normalizeWord(chunk);
    if (highlights.has(normalized) && normalized) {
      return (
        <span key={`${chunk}-${index}`} className="text-gold font-bold">
          {chunk}
        </span>
      );
    }
    return <span key={`${chunk}-${index}`}>{chunk}</span>;
  });
}

export default function SimilarLines({ inputText = "", lines = [] }) {
  const highlightSet = new Set(
    inputText
      .split(/\s+/)
      .map(normalizeWord)
      .filter(Boolean),
  );

  return (
    <div className="grid gap-5 lg:grid-cols-3">
      {lines.map((line, index) => (
        <motion.article
          key={`${line.text.slice(0, 24)}-${index}`}
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.12 * index }}
          className="glass-panel rounded-[28px] p-6"
        >
          <div className="mb-5 flex items-center justify-between gap-4">
            <div>
              <p className="section-eyebrow text-[10px]">{line.poet}</p>
              <p className="mt-1 font-mono text-[10px] uppercase tracking-widest text-text-dim">
                corpus match
              </p>
            </div>
            <div
              className="relative h-14 w-14 rounded-full"
              style={{
                background: `conic-gradient(#C9A84C ${Math.max(line.score, 0.02) * 360}deg, rgba(255,255,255,0.05) 0deg)`,
              }}
            >
              <div className="absolute inset-[5px] flex items-center justify-center rounded-full bg-background font-mono text-[10px] text-gold">
                {line.score.toFixed(3)}
              </div>
            </div>
          </div>
          <blockquote className="font-display text-xl leading-relaxed text-text">
            {renderHighlightedLine(line.text, highlightSet)}
          </blockquote>
        </motion.article>
      ))}
    </div>
  );
}
