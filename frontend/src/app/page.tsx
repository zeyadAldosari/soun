import Image from 'next/image';
import logo from '@/assets/logo.svg';
import HeroSection from '../components/hero-section';
import Background from '../components/background';
import DicomViewer from '@/components/DicomViewer';

export default function Home() {
  return (
    <div dir="rtl">
      <main className="w-screen min-h-screen relative overflow-hidden">
        <Background />
        <header className="relative p-5 z-50">
          <Image priority alt="logo" src={logo} className="w-40" />
        </header>
        <HeroSection />
      </main>
      <section className="w-screen grid place-items-center min-h-screen relative overflow-hidden">
        <div className="p-2 grid place-items-center relative bg-[#404040]/9 border border-[#404040]/10 rounded-2xl bg-opacity-20 backdrop-blur-sm">
          <DicomViewer />
        </div>
      </section>
    </div>
  );
}
