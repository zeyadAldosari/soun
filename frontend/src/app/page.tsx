import Image from "next/image";
import logo from "@/assets/logo.svg";
import HeroSection from "./components/hero-section";


export default function Home() {
    return (
        <div dir="rtl">
            <header className="absolute p-5 z-50">
                <Image priority alt="logo" src={logo} className="w-40" />
            </header>
            <HeroSection/>
        </div>
    );
}
