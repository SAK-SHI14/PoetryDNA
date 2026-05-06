import Plot from "react-plotly.js";

// Bright, saturated, clearly distinct colors for each poet
const POET_COLORS = {
  Shakespeare: "#FF3366",   // hot pink / crimson
  Keats:       "#00E5FF",   // electric cyan
  Wordsworth:  "#76FF03",   // vivid lime green
  Dickinson:   "#FF9100",   // vivid amber orange
  Whitman:     "#D500F9",   // neon magenta/purple
  Frost:       "#00E676",   // bright emerald
};

export default function EmbeddingPlot3D({ points = [], queryPoint = null, nearestNeighbors = [] }) {
  if (!points.length) {
    return (
      <div className="flex h-[400px] items-center justify-center rounded-[28px] border border-white/5 bg-black/20">
        <p className="font-mono text-[10px] uppercase tracking-widest text-text-dim">
          Loading embedding field…
        </p>
      </div>
    );
  }

  const nnTexts = new Set(nearestNeighbors.map((n) => n.text ?? n.line ?? ""));
  const neighborIndices = new Set();

  // Group corpus points by poet
  const byPoet = {};
  points.forEach((p, i) => {
    const poet = p.poet ?? p.label ?? "Unknown";
    if (!byPoet[poet]) byPoet[poet] = [];
    byPoet[poet].push({ ...p, i });
    if (nnTexts.has(p.text ?? "")) neighborIndices.add(i);
  });

  const traces = Object.entries(byPoet).map(([poet, pts]) => ({
    type: "scatter3d",
    name: poet,
    x: pts.map((p) => p.x ?? (p.coords?.[0] ?? 0)),
    y: pts.map((p) => p.y ?? (p.coords?.[1] ?? 0)),
    z: pts.map((p) => p.z ?? (p.coords?.[2] ?? 0)),
    mode: "markers",
    text: pts.map((p) => `<b>${poet}</b><br>${(p.text ?? "").slice(0, 60)}…`),
    hovertemplate: "%{text}<extra></extra>",
    marker: {
      size: pts.map((p) => (neighborIndices.has(p.i) ? 10 : 5)),
      color: POET_COLORS[poet] ?? "#FFFFFF",
      opacity: pts.map((p) => (neighborIndices.has(p.i) ? 1.0 : 0.82)),
      symbol: "circle",
    },
  }));

  // Query point (your input)
  if (queryPoint) {
    traces.push({
      type: "scatter3d",
      name: "Your verse",
      x: [queryPoint[0] ?? queryPoint.x ?? 0],
      y: [queryPoint[1] ?? queryPoint.y ?? 0],
      z: [queryPoint[2] ?? queryPoint.z ?? 0],
      mode: "markers+text",
      text: ["◆ Your verse"],
      textposition: "top center",
      marker: { size: 10, color: "#FFFFFF", symbol: "diamond", opacity: 1 },
    });
  }

  return (
    <div className="plot-frame h-[480px] w-full rounded-[28px] p-2">
      <Plot
        data={traces}
        layout={{
          autosize: true,
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: "#6B6560", family: "DM Mono", size: 10 },
          margin: { l: 0, r: 0, t: 0, b: 0 },
          legend: { x: 0, y: 1, font: { color: "#E8E0D0" } },
          scene: {
            xaxis: { gridcolor: "rgba(255,255,255,0.04)", zerolinecolor: "rgba(255,255,255,0.08)", showticklabels: false },
            yaxis: { gridcolor: "rgba(255,255,255,0.04)", zerolinecolor: "rgba(255,255,255,0.08)", showticklabels: false },
            zaxis: { gridcolor: "rgba(255,255,255,0.04)", zerolinecolor: "rgba(255,255,255,0.08)", showticklabels: false },
            camera: { eye: { x: 1.4, y: 1.4, z: 1.2 } },
            bgcolor: "rgba(0,0,0,0)",
          },
        }}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
