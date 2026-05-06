export const POET_METADATA = {
  Shakespeare: {
    era: "Elizabethan · 1564–1616",
    color: "#1E3A8A",
    styleTags: [
      "Dramatic rhetoric",
      "Volta-rich sonnet turns",
      "Psychological intensity",
      "Elevated metaphor",
    ],
  },
  Keats: {
    era: "Romantic · 1795–1821",
    color: "#065F46",
    styleTags: [
      "Sensuous imagery",
      "Lush musicality",
      "Mythic atmosphere",
      "Meditative longing",
    ],
  },
  Milton: {
    era: "Restoration · 1608–1674",
    color: "#4C1D95",
    styleTags: [
      "Epic cadence",
      "Biblical grandeur",
      "Elevated syntax",
      "Moral cosmology",
    ],
  },
  Tennyson: {
    era: "Victorian · 1809–1892",
    color: "#92400E",
    styleTags: [
      "Melancholic tone",
      "Musical rhythm",
      "Nature imagery",
      "Elegiac mood",
    ],
  },
  Coleridge: {
    era: "Romantic · 1772–1834",
    color: "#134E4A",
    styleTags: [
      "Dreamlike surrealism",
      "Mystic symbolism",
      "Conversational lyricism",
      "Philosophical wonder",
    ],
  },
  Wordsworth: {
    era: "Romantic · 1770–1850",
    color: "#881337",
    styleTags: [
      "Pastoral reflection",
      "Plainspoken diction",
      "Moral introspection",
      "Landscape memory",
    ],
  },
};

export const PROJECT_META = {
  accuracy: 88.5,
  trainingSamples: 2495,
  modelCount: 3,
  githubUrl: "https://github.com/MehtabSingh3711/PoetryDNA",
  team: [
    {
      name: "Mehtab Singh",
      role: "DistilBERT deep-learning pipeline, visual experience, and full-stack integration.",
    },
    {
      name: "Sakshi Verma",
      role: "LightGBM linguistic classifier, stylometric feature engineering, and fusion calibration.",
    },
  ],
};

export const SAMPLE_TEXT = {
  identify: `Half a league, half a league,\nHalf a league onward,\nAll in the valley of Death\nRode the six hundred.`,
  distilbert: `Shall I compare thee to a summer's day?\nThou art more lovely and more temperate.`,
  sbert: `In Xanadu did Kubla Khan\nA stately pleasure-dome decree:\nWhere Alph, the sacred river, ran\nThrough caverns measureless to man`,
  models: `Shall I compare thee to a summer's day?\nThou art more lovely and more temperate.`,
  nlp: `Season of mists and mellow fruitfulness,\nClose bosom-friend of the maturing sun;`,
};
