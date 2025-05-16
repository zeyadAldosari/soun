"use client";

import glowPurple from "@/assets/glow-purple.svg";
import glowBlue from "@/assets/glow-blue.svg";
import blackGlow from "@/assets/glow-black.svg";
import Image from "next/image";
import gridLines from "@/assets/grid-lines.svg";
import gridLines2 from "@/assets/grid-lines-clear.png";
import hand from "@/assets/hand.png";
import skull from "@/assets/skull.png";
import { motion, Variants } from "motion/react";

export default function HeroSection() {
    const textVariants: Variants = {
        hidden: {
            opacity: 0,
            y: 20,
        },
        "text-pop-in": {
            opacity: 1,
            y: 0,
            transition: {
                type: "spring",
                stiffness: 100,
                damping: 20,
            },
        },
    };

    return (
        <main className="w-screen min-h-screen flex justify-center items-center flex-col relative overflow-hidden">
            <Image
                alt="grid lines"
                src={gridLines}
                priority
                className="w-full h-full absolute object-cover opacity-20 -z-30 top-0"
            />
            <motion.div
                animate={{ opacity: 1 }}
                initial={{ opacity: 0 }}
                transition={{ duration: 1 }}
            >
                <Image
                    priority
                    alt="blue glow"
                    src={glowBlue}
                    className="absolute top-0 left-0 sm:block hidden -z-10"
                />
                <Image
                    alt="blue glow"
                    src={blackGlow}
                    priority
                    className="absolute top-0 left-0 sm:block hidden -z-20"
                />
                <Image
                    alt="purple glow"
                    priority
                    src={glowPurple}
                    className="absolute top-0 left-30 sm:block hidden -z-10"
                />
            </motion.div>

            <div className="flex justify-around h-full items-center lg:flex-row flex-col-reverse gap-10 z-10 px-5 pt-10 w-full">
                <div className="grid gap-4">
                    <motion.div
                        transition={{ staggerChildren: 0.2 }}
                        initial="hidden"
                        animate={"text-pop-in"}
                        className="grid gap-4"
                    >
                        <motion.h1
                            className="xl:text-8xl text-6xl w-f font-bold text-white sm:text-nowrap"
                            variants={textVariants}
                        >
                            بياناتك في الحفظ <br />
                            <motion.span
                                animate={{
                                    backgroundPositionX: [0, "100%", 0],
                                }}
                                transition={{
                                    repeat: Infinity,
                                    duration: 5,
                                }}
                                className=" bg-gradient-to-r from-[#002DC9] to-[#8E00D5] bg-clip-text text-transparent bg-size-[200%] font-bold"
                            >
                                <motion.span
                                    animate={{ opacity: 0 }}
                                    transition={{ delay: 1.5 }}
                                    className="absolute z-10 text-white"
                                >
                                    و الصون
                                </motion.span>
                                و الصون
                            </motion.span>
                        </motion.h1>
                        <motion.p
                            variants={textVariants}
                            className="xl:text-xl text-lg max-w-2xl text-white mt-4"
                        >
                            برنامج يهدف إلى حماية خصوصية المرضى عبر تحويل
                            بياناتهم الصحية إلى بيانات مجهولة الهوية.
                        </motion.p>
                        <motion.a
                            variants={textVariants}
                            href=""
                            className="text-white overflow-hidden relative xl:text-xl text-lg font-bold bg-black rounded-2xl px-8 py-4 w-min text-nowrap hover:bg-[#050505] transition-colors"
                        >
                            <Image
                                alt="grid lines"
                                src={gridLines2}
                                priority
                                className="absolute top-0 left-0 opacity-20 z-10 scale-200"
                            />
                            جرب الآن
                        </motion.a>
                    </motion.div>
                </div>
                <motion.div
                    className="p-5 max-w-[600px] w-full relative bg-[#404040]/9 border border-[#404040]/10 rounded-2xl bg-opacity-20 backdrop-blur-sm"
                >
                    <motion.div
                        animate="text-pop-in"
                        initial="hidden"
                        transition={{ staggerChildren: 0.2 }}
                        className="w-full h-full relative"
                    >
                        <motion.div
                            animate={{ height: 0 }}
                            transition={{
                                duration: 2,
                                delay: 3,
                                type: "linear",
                            }}
                            className="absolute w-full h-full bottom-0 flex flex-col"
                        >
                            <motion.div
                                initial={{
                                    opacity: 0,
                                }}
                                animate={{
                                    opacity: 1,
                                }}
                                transition={{
                                    opacity: { duration: 0.5, delay: 3 },
                                }}
                                className="w-full h-10 grid"
                            >
                                <div className="bg-gradient-to-b from-transparent to-black w-full" />
                                <div className="bg-gradient-to-b from-black  via-50% via-[#4A006F] to-black w-full" />
                                <div className="bg-gradient-to-b from-black to-transparent w-full" />
                            </motion.div>
                            <div className="w-full flex-1 overflow-hidden relative">
                                <motion.div
                                    animate="text-pop-in"
                                    initial="hidden"
                                    transition={{
                                        staggerChildren: 0.2,
                                        delayChildren: 1,
                                    }}
                                    dir="ltr"
                                    className="absolute bottom-2 right-2 text-white font-bold text-md"
                                >
                                    <motion.div variants={textVariants}>
                                        Name: Mohammed Salman
                                    </motion.div>
                                    <motion.div variants={textVariants}>
                                        Age: 25
                                    </motion.div>
                                    <motion.div variants={textVariants}>
                                        ID: 123456789
                                    </motion.div>
                                </motion.div>
                            </div>
                        </motion.div>
                        <motion.div variants={textVariants}>
                            <Image
                                alt="hands"
                                src={hand}
                                className="rounded-2xl w-full"
                            />
                        </motion.div>
                        <motion.div
                            variants={textVariants}
                            className="p-5 w-1/2 aspect-square -left-20 top-3/5 absolute bg-[#404040]/9 border border-[#404040]/10 rounded-2xl bg-opacity-20 backdrop-blur-md"
                        >
                            <Image
                                alt="hands"
                                src={skull}
                                fill
                                className="rounded-2xl p-5"
                            />
                        </motion.div>
                    </motion.div>
                </motion.div>
            </div>
        </main>
    );
}
