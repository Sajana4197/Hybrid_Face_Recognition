// tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "sans-serif"],
      },
      keyframes: {
        "red-glow": {
          "0%, 100%": { boxShadow: "0 0 0px 0px rgba(239, 68, 68, 0.0)" },
          "50%": { boxShadow: "0 0 25px 8px rgba(239, 68, 68, 0.7)" },
        },
        "green-glow": {
          "0%, 100%": { boxShadow: "0 0 0px 0px rgba(74, 222, 128, 0.0)" },
          "50%": { boxShadow: "0 0 25px 8px rgba(74, 222, 128, 0.6)" },
        },
        // --- ADD THIS NEW KEYFRAME ---
        "gradient-move": {
          "0%, 100%": { "background-position": "0% 50%" },
          "50%": { "background-position": "100% 50%" },
        },
      },
      animation: {
        "red-glow": "red-glow 1.2s ease-out infinite",
        "green-glow": "green-glow 1.2s ease-out infinite",
        // --- ADD THIS NEW ANIMATION ---
        "gradient-move": "gradient-move 4s ease infinite",
      },
      // --- ADD THIS NEW PROPERTY ---
      backgroundSize: {
        "300%": "300% 300%",
      },
    },
  },
  plugins: [],
};
