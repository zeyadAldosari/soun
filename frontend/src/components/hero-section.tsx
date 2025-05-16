'use client';

import Image from 'next/image';
import gridLines2 from '@/assets/grid-lines-clear.png';
import hand from '@/assets/hand.png';
import skull from '@/assets/skull.png';
import { keyframes, motion, Variants } from 'motion/react';

export default function HeroSection() {
  const textVariants: Variants = {
    hidden: {
      opacity: 0,
      y: 20,
    },
    'text-pop-in': {
      opacity: 1,
      y: 0,
      transition: {
        type: 'spring',
        stiffness: 100,
        damping: 20,
      },
    },
  };

  return (
    <div className="flex justify-center h-full items-center lg:flex-row flex-col-reverse lg:gap-10 gap-30 z-10 px-5 pt-10 w-full">
      <div className="grid gap-4">
        <motion.div
          transition={{ staggerChildren: 0.2 }}
          initial="hidden"
          animate={'text-pop-in'}
          className="grid gap-4  p-2"
        >
          <motion.h1
            className="xl:text-8xl text-6xl font-bold text-white sm:text-nowrap"
            variants={textVariants}
          >
            بياناتك في الحفظ <br />
            <motion.span
              animate={{
                backgroundPositionX: [0, '100%', 0],
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
            className="xl:text-xl text-md max-w-2xl text-white mt-4"
          >
            برنامج يهدف إلى حماية خصوصية المرضى عبر تحويل بياناتهم الصحية إلى
            بيانات مجهولة الهوية.
          </motion.p>
          <motion.a
            variants={textVariants}
            href=""
            className="text-white overflow-hidden relative xl:text-xl text-md font-bold bg-black rounded-2xl px-8 py-4 w-min text-nowrap hover:bg-[#050505] transition-colors"
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
      <div className="pe-20 lg:w-full w-full lg:max-w-[600px] max-w-[400px]">
        <motion.div className="p-2 w-full relative bg-[#404040]/9 border border-[#404040]/10 rounded-2xl bg-opacity-20 backdrop-blur-sm">
          <motion.div
            animate="text-pop-in"
            initial="hidden"
            transition={{ staggerChildren: 0.2 }}
            className="w-full h-full relative"
          >
            <motion.div
              animate={{ height: ['calc(100% + 0px)', 'calc(0% + 40px)'] }}
              transition={{
                duration: 2,
                delay: 2,
                ease: 'easeInOut',
                repeat: Infinity,
                repeatDelay: 0.5,
                repeatType: 'mirror',
              }}
              className="absolute w-full bottom-0 flex flex-col"
            >
              <motion.div
              initial={{ opacity: 0 }}
                animate={{
                  opacity: [0, 1],
                }}
                transition={{
                  opacity: { duration: 0.5, delay: 2, repeatDelay: Infinity },
                }}
                className="w-full h-10 grid"
              >
                <div className="bg-gradient-to-b from-transparent to-black w-full" />
                <div className="bg-gradient-to-b from-black  via-50% via-[#4A006F] to-black w-full" />
                <div className="bg-gradient-to-b from-black to-transparent w-full" />
              </motion.div>
              <div className="w-full flex-1 overflow-hidden relative rounded-2xl">
                <motion.div
                  animate="text-pop-in"
                  initial="hidden"
                  transition={{
                    staggerChildren: 0.2,
                    delayChildren: 1,
                  }}
                  dir="ltr"
                  className="absolute bottom-2 right-2 text-white font-bold lg:text-[13px] text-[8px]"
                >
                  <motion.div variants={textVariants}>
                    Name: Mohammed Salman
                  </motion.div>
                  <motion.div variants={textVariants}>Age: 25</motion.div>
                  <motion.div variants={textVariants}>ID: 123456789</motion.div>
                </motion.div>
              </div>
            </motion.div>
            <motion.div variants={textVariants}>
              <Image alt="hands" src={hand} className="rounded-2xl w-full" />
            </motion.div>
            <motion.div
              variants={textVariants}
              className="w-1/2 aspect-square -left-20 top-3/5 absolute bg-[#404040]/9 border border-[#404040]/10 rounded-2xl bg-opacity-20 backdrop-blur-md"
            >
              <Image alt="hands" src={skull} fill className="rounded-2xl p-2" />
            </motion.div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
}
