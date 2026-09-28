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
          bgPrimary: "#0F172A",     // Deep Slate base layer
          bgSecondary: "#1E293B",   // Charcoal surface layer
          textPrimary: "#F8FAFC",   // High-contrast Off-White
          textMuted: "#94A3B8",     // Mid-tone slate metadata
          accentPrimary: "#6366F1", // Electric Indigo
          accentSecondary: "#14B8A6",// Vibrant Teal
          border: "#334155",        // Subtle structural boundary
        },
      },
      fontFamily: {
        sans: ["Inter", "Roboto", "system-ui", "sans-serif"],
        display: ["Plus Jakarta Sans", "Inter", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      boxShadow: {
        'samvaad-soft': '0px 4px 20px rgba(0, 0, 0, 0.25)',
      },
      borderRadius: {
        'control': '8px',
        'card': '12px',
      },
      backdropBlur: {
        'glass': '12px',
      }
    },
  },
  plugins: [],
}
