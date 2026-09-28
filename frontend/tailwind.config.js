/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        samvaad: {
          bgPrimary: "#09090B",     // Pure dark background
          bgSecondary: "#18181B",   // Slightly lighter dark card
          textPrimary: "#FAFAFA",   // Off-white text
          textMuted: "#A1A1AA",     // Zinc-400
          accentPrimary: "#3B82F6", // Bright blue
          accentSecondary: "#8B5CF6",// Purple
          border: "#27272A",        // Zinc-800
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        display: ["Plus Jakarta Sans", "Inter", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      boxShadow: {
        'glow': '0px 0px 20px rgba(59, 130, 246, 0.15)',
        'glow-hover': '0px 0px 30px rgba(59, 130, 246, 0.25)',
      },
      borderRadius: {
        'card': '16px',
        'button': '12px',
      }
    },
  },
  plugins: [],
}
