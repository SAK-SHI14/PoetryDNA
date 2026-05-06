import { useEffect, useState } from "react";

export default function useTypewriter(text, speed = 70) {
  const [visible, setVisible] = useState("");

  useEffect(() => {
    let index = 0;
    setVisible("");
    const timer = window.setInterval(() => {
      index += 1;
      setVisible(text.slice(0, index));
      if (index >= text.length) {
        window.clearInterval(timer);
      }
    }, speed);

    return () => window.clearInterval(timer);
  }, [text, speed]);

  return visible;
}
