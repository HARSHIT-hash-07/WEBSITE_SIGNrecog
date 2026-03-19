import type { Metadata } from "next";
import { Lexend_Deca, LXGW_WenKai_Mono_TC } from "next/font/google";
import "./globals.css";
import { Header } from "@/components/layout/Header";

const lexendDeca = Lexend_Deca({
  variable: "--font-heading",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800", "900"],
});

const lxgwWenKai = LXGW_WenKai_Mono_TC({
  variable: "--font-body",
  subsets: ["latin"],
  weight: ["400"],
});

export const metadata: Metadata = {
  title: "SignBridge",
  description: "Real-time Text-to-Sign Language Translation",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${lexendDeca.variable} ${lxgwWenKai.variable} antialiased transition-colors duration-300`}
      >
        <Header />
        <main className="pt-16 min-h-screen">{children}</main>
      </body>
    </html>
  );
}
