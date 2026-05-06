import { motion } from "framer-motion";
import { NavLink } from "react-router-dom";

export default function SideNavigation({ routes = [] }) {
  return (
    <nav
      className="fixed left-4 z-50"
      style={{ top: "50%", transform: "translateY(-50%)" }}
    >
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.6, delay: 0.4 }}
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "0.35rem",
          padding: "0.65rem",
          borderRadius: "1rem",
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,255,255,0.06)",
        }}
      >
        {routes.map((route) => (
          <NavLink
            key={route.path}
            to={route.path}
            className={({ isActive }) =>
              `side-nav-item ${isActive ? "active" : ""}`
            }
            title={route.label}
          >
            <span className="side-nav-dot" />
          </NavLink>
        ))}
      </motion.div>

      <style>{`
        .side-nav-item {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 28px;
          height: 28px;
          border-radius: 8px;
          transition: background 0.25s ease;
          position: relative;
        }
        .side-nav-item:hover {
          background: rgba(255,255,255,0.06);
        }
        .side-nav-item:hover::after {
          content: attr(title);
          position: absolute;
          left: 36px;
          white-space: nowrap;
          font-family: "DM Mono", monospace;
          font-size: 10px;
          text-transform: uppercase;
          letter-spacing: 0.16em;
          color: #C9A84C;
          background: rgba(13, 12, 10, 0.92);
          border: 1px solid rgba(201, 168, 76, 0.2);
          padding: 4px 10px;
          border-radius: 6px;
          pointer-events: none;
        }
        .side-nav-dot {
          width: 6px;
          height: 6px;
          border-radius: 999px;
          background: rgba(255,255,255,0.18);
          transition: all 0.3s ease;
        }
        .side-nav-item.active .side-nav-dot {
          background: #C9A84C;
          box-shadow: 0 0 10px rgba(201, 168, 76, 0.4);
          width: 8px;
          height: 8px;
        }
      `}</style>
    </nav>
  );
}
