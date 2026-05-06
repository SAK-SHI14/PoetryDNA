import { motion } from "framer-motion";

export default function ConfidenceBars({ items = [] }) {
  return (
    <div className="space-y-6">
      {items.map((item, index) => {
        // Backend returns different keys in different endpoints:
        // /predict  → top_poets has { poet, prob }      (already %)
        // /explain  → top_poets has { poet, probability } (0-1 fraction)
        // fallback  → item may have { poet, confidence }  (already %)
        const rawValue =
          item.confidence ?? item.prob ?? (item.probability != null ? item.probability * 100 : 0);
        const confidence = typeof rawValue === "number" && isFinite(rawValue) ? rawValue : 0;

        return (
          <div key={item.poet ?? index} className="space-y-3">
            <div className="flex items-center justify-between font-mono text-xs uppercase tracking-widest">
              <span className={index === 0 ? "text-gold" : "text-text-dim"}>{item.poet}</span>
              <span className="text-text-dim">{confidence.toFixed(1)}%</span>
            </div>
            <div className="h-[6px] w-full overflow-hidden rounded-full bg-white/5">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${confidence}%` }}
                transition={{ duration: 1, delay: index * 0.1, ease: [0.22, 1, 0.36, 1] }}
                className={`h-full rounded-full ${
                  index === 0
                    ? "bg-gradient-to-r from-gold to-crimson shadow-[0_0_20px_rgba(201,168,76,0.3)]"
                    : "bg-white/10"
                }`}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
