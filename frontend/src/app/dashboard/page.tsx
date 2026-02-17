'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { AudioUploader } from '@/components/audio/AudioUploader';
import { AudioList } from '@/components/audio/AudioList';
import { StemSeparation } from '@/components/audio/StemSeparation';
import { MoodClassifier } from '@/components/audio/MoodClassifier';
import { MusicGenerator } from '@/components/audio/MusicGenerator';
import { useAuthStore } from '@/lib/store';
import { Upload, Scissors, Smile, Wand2, Music, LogOut } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { motion } from 'framer-motion';
import { Toaster } from 'react-hot-toast';

export default function DashboardPage() {
  const router = useRouter();
  const { isAuthenticated, user, logout } = useAuthStore();

  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/login');
    }
  }, [isAuthenticated, router]);

  const handleLogout = () => {
    logout();
    router.push('/');
  };

  if (!isAuthenticated) {
    return null;
  }

  return (
    <div className="min-h-screen bg-background">
      <Toaster position="top-right" />
      
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Music className="w-8 h-8 text-primary" />
              <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
                SonicAI
              </h1>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="text-sm text-muted-foreground">
                Welcome, <span className="text-foreground font-medium">{user?.username}</span>
              </div>
              <Button variant="outline" size="sm" onClick={handleLogout}>
                <LogOut className="w-4 h-4 mr-2" />
                Logout
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Tabs defaultValue="upload" className="w-full">
            <TabsList className="grid w-full grid-cols-5 mb-8">
              <TabsTrigger value="upload" className="flex items-center gap-2">
                <Upload className="w-4 h-4" />
                Upload
              </TabsTrigger>
              <TabsTrigger value="library" className="flex items-center gap-2">
                <Music className="w-4 h-4" />
                Library
              </TabsTrigger>
              <TabsTrigger value="stems" className="flex items-center gap-2">
                <Scissors className="w-4 h-4" />
                Stems
              </TabsTrigger>
              <TabsTrigger value="mood" className="flex items-center gap-2">
                <Smile className="w-4 h-4" />
                Mood
              </TabsTrigger>
              <TabsTrigger value="generate" className="flex items-center gap-2">
                <Wand2 className="w-4 h-4" />
                Generate
              </TabsTrigger>
            </TabsList>

            <TabsContent value="upload" className="space-y-6">
              <div className="max-w-3xl mx-auto">
                <h2 className="text-3xl font-bold text-foreground mb-2">
                  Upload Audio
                </h2>
                <p className="text-muted-foreground mb-8">
                  Upload your audio files to get started with AI-powered processing
                </p>
                <AudioUploader />
              </div>
            </TabsContent>

            <TabsContent value="library" className="space-y-6">
              <div>
                <h2 className="text-3xl font-bold text-foreground mb-2">
                  Audio Library
                </h2>
                <p className="text-muted-foreground mb-8">
                  Manage and organize your uploaded audio files
                </p>
                <AudioList />
              </div>
            </TabsContent>

            <TabsContent value="stems" className="space-y-6">
              <div className="max-w-4xl mx-auto">
                <h2 className="text-3xl font-bold text-foreground mb-2">
                  Stem Separation
                </h2>
                <p className="text-muted-foreground mb-8">
                  Separate your audio into individual stems using AI
                </p>
                <StemSeparation />
              </div>
            </TabsContent>

            <TabsContent value="mood" className="space-y-6">
              <div className="max-w-4xl mx-auto">
                <h2 className="text-3xl font-bold text-foreground mb-2">
                  Mood Classification
                </h2>
                <p className="text-muted-foreground mb-8">
                  Analyze the emotional content and characteristics of your music
                </p>
                <MoodClassifier />
              </div>
            </TabsContent>

            <TabsContent value="generate" className="space-y-6">
              <div className="max-w-4xl mx-auto">
                <h2 className="text-3xl font-bold text-foreground mb-2">
                  Music Generation
                </h2>
                <p className="text-muted-foreground mb-8">
                  Create original music from text descriptions using AI
                </p>
                <MusicGenerator />
              </div>
            </TabsContent>
          </Tabs>
        </motion.div>
      </main>
    </div>
  );
}
