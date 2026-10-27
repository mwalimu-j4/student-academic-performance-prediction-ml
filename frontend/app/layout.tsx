import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Student Outcome Intelligence",
  description: "Transparent admission-time student outcome predictions",
};
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
