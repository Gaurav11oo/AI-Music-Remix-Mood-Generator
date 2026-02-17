'use client';

import { motion } from 'framer-motion';
import { Music, Waves, Sparkles, Zap, Headphones, Wand2 } from 'lucide-react';
import Link from 'next/link';

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-slate-900 text-white overflow-hidden">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <motion.div
          className="absolute -top-1/2 -right-1/2 w-full h-full bg-purple-500/10 rounded-full blur-3xl"
          animate={{
            scale: [1, 1.2, 1],
            rotate: [0, 90, 0],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: "linear"
          }}
        />
        <motion.div
          className="absolute -bottom-1/2 -left-1/2 w-full h-full bg-pink-500/10 rounded-full blur-3xl"
          animate={{
            scale: [1.2, 1, 1.2],
            rotate: [90, 0, 90],
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            ease: "linear"
          }}
        />
      </div>

      {/* Navigation */}
      <nav className="relative z-10 container mx-auto px-6 py-8 flex justify-between items-center">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex items-center gap-2 text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent"
        >
          <Music className="w-8 h-8 text-purple-400" />
          SonicAI
        </motion.div>
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex gap-4"
        >
          <Link href="/login">
            <button className="px-6 py-2 text-sm font-medium text-purple-300 hover:text-white transition-colors">
              Sign In
            </button>
          </Link>
          <Link href="/register">
            <button className="px-6 py-2 text-sm font-medium bg-gradient-to-r from-purple-500 to-pink-500 rounded-full hover:shadow-lg hover:shadow-purple-500/50 transition-all">
              Get Started
            </button>
          </Link>
        </motion.div>
      </nav>

      {/* Hero Section */}
      <section className="relative z-10 container mx-auto px-6 py-20 text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div className="inline-block mb-6 px-4 py-2 bg-purple-500/20 border border-purple-500/30 rounded-full text-sm backdrop-blur-sm">
            <Sparkles className="inline w-4 h-4 mr-2" />
            Powered by Advanced AI
          </div>
          
          <h1 className="text-7xl md:text-8xl font-black mb-6 leading-tight">
            <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-purple-400 bg-clip-text text-transparent animate-gradient">
              Transform
            </span>
            <br />
            <span className="text-white">Your Music</span>
          </h1>
          
          <p className="text-xl md:text-2xl text-purple-200 max-w-3xl mx-auto mb-12 leading-relaxed font-light">
            Separate stems, classify moods, remix genres, and generate entirely new music using cutting-edge AI technology.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-16">
            <Link href="/dashboard">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="px-8 py-4 text-lg font-semibold bg-gradient-to-r from-purple-600 to-pink-600 rounded-full shadow-2xl shadow-purple-500/50 hover:shadow-purple-500/70 transition-all flex items-center gap-2 justify-center"
              >
                <Wand2 className="w-5 h-5" />
                Start Creating
              </motion.button>
            </Link>
            <Link href="#features">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="px-8 py-4 text-lg font-semibold bg-white/10 backdrop-blur-sm border border-white/20 rounded-full hover:bg-white/20 transition-all"
              >
                Explore Features
              </motion.button>
            </Link>
          </div>

          {/* Animated waveform visualization */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3 }}
            className="max-w-4xl mx-auto"
          >
            <div className="relative h-32 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-3xl border border-purple-500/30 overflow-hidden backdrop-blur-sm">
              <div className="absolute inset-0 flex items-center justify-center gap-1 px-4">
                {[...Array(50)].map((_, i) => (
                  <motion.div
                    key={i}
                    className="w-1 bg-gradient-to-t from-purple-500 to-pink-500 rounded-full"
                    animate={{
                      height: [
                        `${20 + Math.random() * 60}%`,
                        `${20 + Math.random() * 60}%`,
                        `${20 + Math.random() * 60}%`,
                      ],
                    }}
                    transition={{
                      duration: 1 + Math.random(),
                      repeat: Infinity,
                      ease: "easeInOut",
                      delay: i * 0.02,
                    }}
                  />
                ))}
              </div>
            </div>
          </motion.div>
        </motion.div>
      </section>

      {/* Features Grid */}
      <section id="features" className="relative z-10 container mx-auto px-6 py-20">
        <motion.h2
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="text-5xl font-bold text-center mb-16 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent"
        >
          Powerful AI Features
        </motion.h2>

        <div className="grid md:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ scale: 1.05 }}
              className="relative group"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-pink-500 rounded-3xl blur opacity-25 group-hover:opacity-40 transition-opacity" />
              <div className="relative bg-slate-900/50 backdrop-blur-sm border border-purple-500/30 rounded-3xl p-8 h-full">
                <div className="w-14 h-14 mb-6 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center">
                  <feature.icon className="w-7 h-7 text-white" />
                </div>
                <h3 className="text-2xl font-bold mb-4">{feature.title}</h3>
                <p className="text-purple-200 leading-relaxed">{feature.description}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 container mx-auto px-6 py-12 text-center border-t border-purple-500/20">
        <p className="text-purple-300">
          2024 SonicAI. Transforming music with artificial intelligence.
        </p>
      </footer>
    </div>
  );
}

const features = [
  {
    icon: Waves,
    title: "Stem Separation",
    description: "Isolate vocals, drums, bass, and instruments with AI-powered precision using state-of-the-art Demucs models."
  },
  {
    icon: Sparkles,
    title: "Mood Classification",
    description: "Automatically detect and classify the emotional tone of your music with advanced machine learning algorithms."
  },
  {
    icon: Zap,
    title: "Genre Remix",
    description: "Transform any song into different genres - from jazz to EDM - while maintaining the original essence."
  },
  {
    icon: Wand2,
    title: "Text-to-Music",
    description: "Generate completely new music from text prompts using cutting-edge generative AI models."
  },
  {
    icon: Headphones,
    title: "Audio Effects",
    description: "Apply professional-grade effects, adjust tempo and pitch, and fine-tune your tracks to perfection."
  },
  {
    icon: Music,
    title: "Waveform Analysis",
    description: "Visualize and analyze your audio with interactive waveforms, spectrograms, and feature extraction."
  }
];
