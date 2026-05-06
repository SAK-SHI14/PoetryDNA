import { useEffect, useState } from "react";
import { motion, useSpring } from "framer-motion";

export default function QuillCursor() {
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [isVisible, setIsVisible] = useState(false);

  const springX = useSpring(0, { stiffness: 250, damping: 25 });
  const springY = useSpring(0, { stiffness: 250, damping: 25 });

  useEffect(() => {
    const handleMouseMove = (e) => {
      setMousePos({ x: e.clientX, y: e.clientY });
      springX.set(e.clientX);
      springY.set(e.clientY);
      if (!isVisible) setIsVisible(true);
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, [isVisible, springX, springY]);

  if (!isVisible) return null;

  return (
    <>
      {/* Main Dot */}
      <motion.div
        className="pointer-events-none fixed left-0 top-0 z-[9999] h-2 w-2 rounded-full bg-gold shadow-gold-glow"
        style={{ x: mousePos.x - 4, y: mousePos.y - 4 }}
      />
      {/* Lagging Ring */}
      <motion.div
        className="pointer-events-none fixed left-0 top-0 z-[9998] h-10 w-10 rounded-full border border-gold/30"
        style={{ x: springX.get() - 20, y: springY.get() - 20 }}
      />
      {/* Dust particles */}
      {[...Array(3)].map((_, i) => (
        <motion.div
          key={i}
          className="pointer-events-none fixed left-0 top-0 z-[9997] h-1 w-1 rounded-full bg-gold/40"
          animate={{
            x: mousePos.x + (Math.random() - 0.5) * 40,
            y: mousePos.y + (Math.random() - 0.5) * 40,
            scale: [1, 0],
            opacity: [1, 0],
          }}
          transition={{ duration: 0.8, repeat: Infinity, delay: i * 0.2 }}
        />
      ))}
    </>
  );
}
