import { animate, useMotionValue } from "framer-motion";
import { useEffect, useState } from "react";

export default function AnimatedCounter({ value, suffix = "", decimals = 0, className = "" }) {
  const [display, setDisplay] = useState(0);
  const count = useMotionValue(0);

  useEffect(() => {
    const safeValue = typeof value === "number" && Number.isFinite(value) ? value : 0;
    const controls = animate(count, safeValue, {
      duration: 1.6,
      ease: [0.22, 1, 0.36, 1],
      onUpdate: (latest) => setDisplay(latest),
    });
    return () => controls.stop();
  }, [count, value]);

  return (
    <span className={className}>
      {display.toFixed(decimals)}
      {suffix}
    </span>
  );
}
