import { motion } from "framer-motion";

export default function AttentionHeatmap({ attention = [] }) {
  if (!attention || attention.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-x-2 gap-y-3 leading-relaxed">
      {attention.map((wordObj, index) => {
        const score = wordObj.score || 0;
        const normalized = Math.min(score * 12, 1);
        return (
          <motion.span
            key={`${wordObj.word}-${index}`}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: index * 0.012 }}
            className="rounded px-1.5 py-0.5 font-body text-lg transition-colors duration-500"
            style={{
              background: `rgba(201, 168, 76, ${normalized * 0.42})`,
              color: normalized > 0.4 ? "#E8E0D0" : "#E8E0D0",
              borderBottom: `2px solid rgba(201, 168, 76, ${normalized * 0.8})`,
            }}
          >
            {wordObj.word}
          </motion.span>
        );
      })}
    </div>
  );
}
