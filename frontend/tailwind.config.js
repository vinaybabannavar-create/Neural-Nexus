/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#090910",
        surface: "#12121e",
        surfaceHover: "#1c1c2e",
        accent: "#4facfe",
        accentPurple: "#a78bfa",
        glassBorder: "rgba(255, 255, 255, 0.08)",
        trustGreen: "#00f5a0",
        trustYellow: "#ffbd2e",
        trustRed: "#ff5252",
      },
    },
  },
  plugins: [],
};
