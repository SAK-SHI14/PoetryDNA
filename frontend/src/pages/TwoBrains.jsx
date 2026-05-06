import { motion } from "framer-motion";
import { Brain, Cpu, Layers, Zap, BarChart3, GitMerge } from "lucide-react";

const SECTION_ANIM = {
  hidden: { opacity: 0, y: 30 },
  visible: (i) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.12, duration: 0.6, ease: [0.22, 1, 0.36, 1] },
  }),
};

function ModelCard({ icon: Icon, title, subtitle, accuracy, features, accent }) {
  return (
    <motion.div className="glass-panel rounded-[36px] p-8 flex flex-col gap-6" variants={SECTION_ANIM}>
      <div className="flex items-center gap-4">
        <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-white/5 bg-white/5 shadow-gold-soft">
          <Icon size={28} style={{ color: accent }} />
        </div>
        <div>
          <h3 className="font-display text-2xl text-text">{title}</h3>
          <p className="text-sm text-text-dim">{subtitle}</p>
        </div>
      </div>
      <div className="flex items-end gap-3">
        <span className="font-display text-5xl text-gold">{accuracy}%</span>
        <span className="mb-1 text-sm text-text-dim">accuracy</span>
      </div>
      <div className="border-t border-white/5 pt-4">
        <p className="section-eyebrow mb-3 text-[10px]">Key Features</p>
        <ul className="space-y-2">
          {features.map((f) => (
            <li key={f} className="flex items-start gap-2 text-sm text-text-dim leading-relaxed">
              <Zap size={12} className="mt-1 shrink-0 text-gold/60" />
              {f}
            </li>
          ))}
        </ul>
      </div>
    </motion.div>
  );
}

export default function TwoBrains() {
  return (
    <div className="relative z-10 mx-auto max-w-5xl px-6 pb-32 pt-28 sm:px-10">
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7 }}
        className="mb-16 text-center"
      >
        <p className="section-eyebrow mb-4 text-xs">Architecture</p>
        <h1 className="font-display text-5xl leading-tight text-text sm:text-7xl">
          The Two Brains
        </h1>
        <p className="mx-auto mt-6 max-w-2xl text-lg leading-8 text-text-dim">
          PoetryDNA fuses a deep-learning neural network with a hand-crafted
          linguistic classifier. Together, they see what neither could alone.
        </p>
      </motion.div>

      <motion.div className="grid gap-8 md:grid-cols-2" initial="hidden" animate="visible">
        <ModelCard
          icon={Brain}
          title="The Neural Brain"
          subtitle="DistilBERT · Deep Learning"
          accuracy={86.17}
          accent="#C9A84C"
          features={[
            "Pre-trained on 3B+ words of English text",
            "Fine-tuned on 2,495 poetry stanzas",
            "MC Dropout (50 passes) for uncertainty",
            "6 transformer attention layers",
            "Captures semantic & contextual patterns",
          ]}
        />
        <ModelCard
          icon={Cpu}
          title="The Linguistic Brain"
          subtitle="LightGBM · Classical ML"
          accuracy={81.36}
          accent="#8B1A1A"
          features={[
            "32 hand-crafted stylometric features",
            "TF-IDF word n-grams (SVD-150)",
            "Character n-gram morphology (SVD-50)",
            "Prosody: syllables, iambic meter, rhyme",
            "Vocabulary: archaic, nature, divine ratios",
          ]}
        />
      </motion.div>

      {/* Fusion */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4, duration: 0.7 }}
        className="glass-panel mt-12 rounded-[36px] p-8"
      >
        <div className="flex items-center gap-4 mb-6">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-white/5 bg-white/5 shadow-gold-soft">
            <GitMerge size={28} className="text-text" />
          </div>
          <div>
            <h3 className="font-display text-2xl text-text">Weighted Fusion</h3>
            <p className="text-sm text-text-dim">Decision engine</p>
          </div>
        </div>
        <p className="text-text-dim leading-7 mb-8">
          The final prediction is a weighted average of both brains' probability
          distributions. The Neural Brain contributes ~60% and the Linguistic
          Brain ~40%.
        </p>
        <div className="flex flex-col items-center gap-4 sm:flex-row sm:gap-12 justify-center">
          <div className="flex flex-col items-center gap-3">
            <div className="flex h-20 w-20 items-center justify-center rounded-full border border-gold/30 bg-gold/5 shadow-gold-glow">
              <Brain size={32} className="text-gold" />
            </div>
            <span className="font-mono text-xs text-text-dim">60% WEIGHT</span>
          </div>
          <div className="text-3xl text-gold/30">+</div>
          <div className="flex flex-col items-center gap-3">
            <div className="flex h-20 w-20 items-center justify-center rounded-full border border-crimson/30 bg-crimson/5 shadow-crimson-glow">
              <Cpu size={32} className="text-crimson" />
            </div>
            <span className="font-mono text-xs text-text-dim">40% WEIGHT</span>
          </div>
          <div className="text-3xl text-gold/30">=</div>
          <div className="flex flex-col items-center gap-3">
            <div className="flex h-20 w-20 items-center justify-center rounded-full border border-text/30 bg-text/5">
              <Layers size={32} className="text-text" />
            </div>
            <span className="font-mono text-xs text-text-dim">FUSED OUTPUT</span>
          </div>
        </div>
      </motion.div>

      {/* Pipeline */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6, duration: 0.7 }}
        className="glass-panel mt-12 rounded-[36px] p-8"
      >
        <div className="flex items-center gap-4 mb-8">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-white/5 bg-white/5">
            <BarChart3 size={28} className="text-gold" />
          </div>
          <div>
            <h3 className="font-display text-2xl text-text">Processing Pipeline</h3>
            <p className="text-sm text-text-dim">Neural to linguistic flow</p>
          </div>
        </div>
        <div className="space-y-6">
          {[
            { step: "01", label: "Tokenization", desc: "Slicing text into sub-word fragments via DistilBERT's tokenizer." },
            { step: "02", label: "Neural Pass", desc: "50 stochastic forward passes with dropout enabled for confidence calibration." },
            { step: "03", label: "Stylometric Sweep", desc: "Extracting 32 handcrafted linguistic features and TF-IDF representations." },
            { step: "04", label: "Gradient Boosting", desc: "LightGBM classifies the high-dimensional feature vector." },
            { step: "05", label: "Softmax Fusion", desc: "Weighted combination of probability vectors for the final verdict." },
          ].map((item) => (
            <div key={item.step} className="flex gap-6 items-start">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-gold/20 bg-gold/5 font-mono text-xs text-gold">
                {item.step}
              </div>
              <div>
                <p className="text-lg font-display text-text">{item.label}</p>
                <p className="text-sm text-text-dim leading-relaxed">{item.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
