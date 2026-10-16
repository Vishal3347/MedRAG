import type { Config } from "tailwindcss";

const rgb = (name: string) => `rgb(var(--${name}) / <alpha-value>)`;

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: rgb("bg"),
        panel: rgb("panel"),
        ink: rgb("ink"),
        muted: rgb("muted"),
        line: rgb("line"),
        accent: rgb("accent"),
        "accent-soft": rgb("accent-soft"),
        bubble: rgb("bubble"),
        danger: rgb("danger"),
      },
      fontFamily: {
        sans: ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"],
        serif: ["Iowan Old Style", "Charter", "Source Serif Pro", "Georgia", "Cambria", "Times New Roman", "serif"],
      },
      maxWidth: { chat: "800px" },
    },
  },
  plugins: [],
};

export default config;
