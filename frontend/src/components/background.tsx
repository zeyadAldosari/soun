import Image from "next/image";
import gridLines from "@/assets/grid-lines.svg";
import glowPurple from "@/assets/glow-purple.svg";
import glowBlue from "@/assets/glow-blue.svg";
import blackGlow from "@/assets/glow-black.svg";


export default function Background() {
  return (
    <div className="w-screen overflow-hidden h-full">
      <Image
        alt="grid lines"
        src={gridLines}
        priority
        className="w-full h-full absolute object-cover opacity-20 -z-30 top-0"
      />
      <div
      >
        <Image
          priority
          alt="blue glow"
          src={glowBlue}
          className="absolute top-0 left-0 -z-10"
        />
        <Image
          alt="blue glow"
          src={blackGlow}
          priority
          className="absolute top-0 left-0 -z-20"
        />
        <Image
          alt="purple glow"
          priority
          src={glowPurple}
          className="absolute top-0 left-30 -z-10"
        />
      </div>
      <div className="w-full bg-gradient-to-b bg-transparent to-90% to-[#0F1014] bottom-0 absolute h-32"/>
    </div>
  );
}
