/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        background: "#0D0C0A",
        gold: "#C9A84C",
        crimson: "#8B1A1A",
        text: "#E8E0D0",
        "text-dim": "#6B6560",
      },
      fontFamily: {
        display: ["Outfit", "sans-serif"],
        body: ["Inter", "sans-serif"],
        mono: ["DM Mono", "monospace"],
      },
      boxShadow: {
        "gold-glow": "0 0 48px rgba(201, 168, 76, 0.15)",
        "gold-soft": "0 0 20px rgba(201, 168, 76, 0.08)",
        "crimson-glow": "0 0 48px rgba(139, 26, 26, 0.15)",
      },
    },
  },
  plugins: [],
};
