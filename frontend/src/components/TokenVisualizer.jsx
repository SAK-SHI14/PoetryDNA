import { motion } from "framer-motion";

export default function TokenVisualizer({ tokens = [] }) {
  return (
    <div className="flex flex-wrap gap-2">
      {tokens.map((token, index) => (
        <motion.div
          key={`${token}-${index}`}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: index * 0.02 }}
          className="rounded-xl border border-white/5 bg-white/5 px-3 py-1.5 font-mono text-xs text-gold/80"
        >
          {token}
        </motion.div>
      ))}
    </div>
  );
}
