import type { Config } from "tailwindcss";
const config: Config = {
  content: ["./app/**/*.{ts,tsx}"],
  theme: { extend: { colors: { ink: "#152238", teal: "#0f766e" } } },
  plugins: [],
};
export default config;
