import { motion } from "framer-motion";

export default function PageLoader({ label = "Inhaling the verse..." }) {
  return (
    <div className="flex h-screen w-full flex-col items-center justify-center bg-background px-6">
      <div className="relative mb-10 flex h-24 w-24 items-center justify-center">
        <motion.div
          animate={{ rotate: 360, scale: [1, 1.1, 1] }}
          transition={{ duration: 3, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
          className="absolute inset-0 rounded-full border border-gold/20 border-t-gold shadow-gold-glow"
        />
        <div className="h-4 w-4 rounded-full bg-gold shadow-gold-glow" />
      </div>
      <p className="font-mono text-xs uppercase tracking-[0.4em] text-gold/80">{label}</p>
    </div>
  );
}
