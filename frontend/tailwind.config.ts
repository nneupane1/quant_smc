import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: "#07111c",
        bloom: "#091625",
        panel: "#0d1a2a",
        terminal: "#0b1624",
        cyan: "#52d7ff",
        teal: "#2ae6b8",
        amber: "#f6b63c",
        rose: "#ff6b88",
        lime: "#8ef08c",
      },
      boxShadow: {
        bloom: "0 0 0 1px rgba(82, 215, 255, 0.15), 0 22px 60px rgba(0, 0, 0, 0.35)",
        glow: "0 0 30px rgba(82, 215, 255, 0.18)",
        amber: "0 0 30px rgba(246, 182, 60, 0.2)",
      },
      keyframes: {
        floaty: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-6px)" }
        },
        pulseGlow: {
          "0%, 100%": { opacity: "0.55" },
          "50%": { opacity: "1" }
        },
        scan: {
          "0%": { transform: "translateX(-120%)" },
          "100%": { transform: "translateX(120%)" }
        }
      },
      animation: {
        floaty: "floaty 7s ease-in-out infinite",
        "pulse-glow": "pulseGlow 2.8s ease-in-out infinite",
        scan: "scan 3.2s linear infinite"
      },
      backgroundImage: {
        grid: "linear-gradient(rgba(82,215,255,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(82,215,255,0.05) 1px, transparent 1px)"
      }
    },
  },
  plugins: [],
};

export default config;
