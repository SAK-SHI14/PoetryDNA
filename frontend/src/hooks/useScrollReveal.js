import { useEffect } from "react";

/**
 * Reveals [data-reveal] elements inside `ref` as they scroll into view.
 * Uses IntersectionObserver for the animation and MutationObserver to
 * catch elements that are added dynamically (e.g. after API response).
 */
export default function useScrollReveal(ref) {
  useEffect(() => {
    if (!ref.current) return;

    const observed = new Set();

    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const el = entry.target;
            el.style.opacity = "1";
            el.style.transform = "translateY(0)";
            io.unobserve(el); // animate once only
          }
        });
      },
      { threshold: 0.06 }
    );

    function observe(el, index) {
      if (observed.has(el)) return;
      observed.add(el);
      // Set initial state inline (no CSS visibility:hidden needed)
      el.style.opacity = "0";
      el.style.transform = "translateY(28px)";
      el.style.transition = `opacity 0.7s cubic-bezier(0.22,1,0.36,1) ${index * 0.06}s, transform 0.7s cubic-bezier(0.22,1,0.36,1) ${index * 0.06}s`;
      io.observe(el);
    }

    // Observe existing elements
    ref.current.querySelectorAll("[data-reveal]").forEach((el, i) => observe(el, i));

    // Watch for elements added later (results rendered after API call)
    const mo = new MutationObserver(() => {
      if (!ref.current) return;
      const all = ref.current.querySelectorAll("[data-reveal]");
      all.forEach((el, i) => observe(el, i));
    });

    mo.observe(ref.current, { childList: true, subtree: true });

    return () => {
      io.disconnect();
      mo.disconnect();
    };
  }, [ref]);
}
