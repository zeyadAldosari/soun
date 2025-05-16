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

        <DicomViewer/>
        
      </main>
    </div>
  );
}