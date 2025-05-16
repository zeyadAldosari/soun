import Image from "next/image";
import logo from "@/assets/logo.svg";
import HeroSection from "../components/hero-section";
import Background from "../components/background";
import DicomViewer from "@/components/DicomViewer";
import gridLines from "@/assets/grid-lines.svg";
import { DotLottieReact } from "@lottiefiles/dotlottie-react";

export default function Home() {
    return (
        <div dir="rtl">
            <main className="w-screen pb-30 relative overflow-hidden">
                <Background />
                <header className="relative p-5 z-50">
                    <Image priority alt="logo" src={logo} className="w-40" />
                </header>
                <HeroSection />
            </main>
            <section id="upload" className="w-screen pb-10 relative overflow-hidden z-10 bg-[#0F1014]">
                <DicomViewer />
            </section>
            <footer className="h-20 relative ">
                <div className="w-full bg-gradient-to-t from-transparent to-90% to-[#0F1014] top-0 absolute h-10 z-10" />
                <Image
                alt="grid lines"
                src={gridLines}
                priority
                className="w-full h-full object-cover opacity-20 -z-30 top-0"
            />
            
            </footer>
        </div>
    );
}
