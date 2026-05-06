import { AnimatePresence, motion } from "framer-motion";
import { lazy, Suspense, useEffect } from "react";
import { Route, Routes, useLocation } from "react-router-dom";
import PageLoader from "./components/PageLoader";
import ParticleBackground from "./components/ParticleBackground";
import SideNavigation from "./components/SideNavigation";

const Identify = lazy(() => import("./pages/Identify"));
const DistilBERT = lazy(() => import("./pages/DistilBERT"));
const SBERT = lazy(() => import("./pages/SBERT"));
const About = lazy(() => import("./pages/About"));
const TwoBrains = lazy(() => import("./pages/TwoBrains"));
const LinguisticBrain = lazy(() => import("./pages/LinguisticBrain"));

const routes = [
  { path: "/", label: "Identify", poet: "Reveal" },
  { path: "/distilbert", label: "DistilBERT", poet: "Neural" },
  { path: "/sbert", label: "SBERT", poet: "Sim" },
  { path: "/models", label: "The Two Brains", poet: "Fusion" },
  { path: "/nlp", label: "Linguistic Brain", poet: "Features" },
  { path: "/about", label: "About PoetryDNA", poet: "Archive" },
];

function RouteScene({ children }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 22 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -18 }}
      transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
      className="relative z-10"
    >
      {children}
    </motion.div>
  );
}

export default function App() {
  const location = useLocation();

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [location.pathname]);

  return (
    <div className="page-shell min-h-screen overflow-x-hidden bg-background">
      <ParticleBackground />
      <SideNavigation routes={routes} />
      <Suspense fallback={<PageLoader label="Summoning the archive..." />}>
        <AnimatePresence mode="wait">
          <Routes location={location} key={location.pathname}>
            <Route
              path="/"
              element={
                <RouteScene>
                  <Identify />
                </RouteScene>
              }
            />
            <Route
              path="/distilbert"
              element={
                <RouteScene>
                  <DistilBERT />
                </RouteScene>
              }
            />
            <Route
              path="/sbert"
              element={
                <RouteScene>
                  <SBERT />
                </RouteScene>
              }
            />
            <Route
              path="/models"
              element={
                <RouteScene>
                  <TwoBrains />
                </RouteScene>
              }
            />
            <Route
              path="/nlp"
              element={
                <RouteScene>
                  <LinguisticBrain />
                </RouteScene>
              }
            />
            <Route
              path="/about"
              element={
                <RouteScene>
                  <About />
                </RouteScene>
              }
            />
          </Routes>
        </AnimatePresence>
      </Suspense>
    </div>
  );
}
