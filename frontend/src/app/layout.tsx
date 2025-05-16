import type { Metadata } from "next";
import { Rubik } from "next/font/google";
import "./globals.css";

const rubik = Rubik({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "صون",
  description: "برنامج يهدف إلى حماية خصوصية المرضى عبر تحويل بياناتهم الصحية إلى بيانات مجهولة الهوية.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={rubik.className}
      >
        {children}
      </body>
    </html>
  );
}
