/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        mono: {
          bg: "#FAFAFA",         // Shiny White Background
          surface: "#FFFFFF",    // Pure White Surface
          textMain: "#111827",   // Almost Black Text
          textMuted: "#6B7280",  // Slate Gray Muted
          accent: "#000000",     // Pure Black Accent
          border: "#E5E7EB",     // Light border
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        display: ["Plus Jakarta Sans", "Inter", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      boxShadow: {
        'shiny': '0px 10px 30px rgba(0, 0, 0, 0.05)',
        'shiny-hover': '0px 20px 40px rgba(0, 0, 0, 0.08)',
      },
      borderRadius: {
        'card': '16px',
        'button': '9999px', // Pill shape for sleekness
      }
    },
  },
  plugins: [],
}
