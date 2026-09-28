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
          bgPrimary: "#FFFFFF",     // Clean White Background
          bgSecondary: "#F4F4F5",   // Light Gray Surface
          textPrimary: "#000000",   // Solid Black Text
          textMuted: "#52525B",     // Dim Gray
          accentPrimary: "#EAB308", // Shiny Yellow Accent!
          accentSecondary: "#FDE047",// Bright Yellow glow
          border: "#E4E4E7",        // Light border
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        display: ["Plus Jakarta Sans", "Inter", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      boxShadow: {
        'glow': '0px 0px 20px rgba(234, 179, 8, 0.25)',
        'glow-hover': '0px 0px 30px rgba(234, 179, 8, 0.45)',
      },
      borderRadius: {
        'card': '20px',
        'button': '12px',
      }
    },
  },
  plugins: [],
}
