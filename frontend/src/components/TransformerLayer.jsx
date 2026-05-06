import { motion } from "framer-motion";

export default function TransformerLayer({ layer }) {
  const tokens = layer.tokens || [];
  const matrix = layer.matrix || [];

  return (
    <div className="glass-panel rounded-[36px] p-8">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <p className="section-eyebrow text-[10px]">Layer {layer.layer}</p>
          <h3 className="mt-1 font-display text-2xl text-text">Self-Attention</h3>
        </div>
        <div className="flex items-center gap-4 font-mono text-[10px] uppercase tracking-widest text-text-dim">
          <span>Head averaged</span>
          <div className="h-1 w-12 rounded-full bg-gradient-to-r from-transparent to-gold" />
        </div>
      </div>

      <div className="relative mt-8 grid gap-4 lg:grid-cols-[1fr_auto_1fr]">
        <div className="space-y-1">
          {tokens.map((t, i) => (
            <div key={`source-${i}`} className="flex h-6 items-center justify-end px-3 font-mono text-[10px] text-text-dim">
              {t}
            </div>
          ))}
        </div>

        <div className="relative h-full w-[400px]">
          <svg className="h-full w-full overflow-visible">
            {matrix.map((row, i) =>
              row.map((val, j) => {
                if (val < 0.05) return null;
                return (
                  <motion.path
                    key={`attention-${i}-${j}`}
                    d={`M 0,${i * 24 + 12} C 200,${i * 24 + 12} 200,${j * 24 + 12} 400,${j * 24 + 12}`}
                    stroke="#C9A84C"
                    strokeWidth={val * 4}
                    strokeOpacity={val * 0.6}
                    fill="none"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 1, delay: i * 0.02 }}
                  />
                );
              }),
            )}
          </svg>
        </div>

        <div className="space-y-1">
          {tokens.map((t, i) => (
            <div key={`target-${i}`} className="flex h-6 items-center px-3 font-mono text-[10px] text-text-dim">
              {t}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
