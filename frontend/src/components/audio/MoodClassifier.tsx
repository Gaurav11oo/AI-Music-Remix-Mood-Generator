'use client';

import { useState, useEffect } from 'react';
import { Smile, Loader2, TrendingUp } from 'lucide-react';
import { motion } from 'framer-motion';
import { moodAPI } from '@/lib/api';
import { useAudioStore } from '@/lib/store';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import toast from 'react-hot-toast';

const moodEmojis: Record<string, string> = {
  happy: '😊',
  sad: '😢',
  energetic: '⚡',
  calm: '😌',
  angry: '😠',
  romantic: '❤️',
  dark: '🌑',
  uplifting: '🌟'
};

const moodColors: Record<string, string> = {
  happy: 'bg-yellow-500',
  sad: 'bg-blue-500',
  energetic: 'bg-red-500',
  calm: 'bg-green-500',
  angry: 'bg-orange-500',
  romantic: 'bg-pink-500',
  dark: 'bg-purple-500',
  uplifting: 'bg-cyan-500'
};

export function MoodClassifier() {
  const { selectedAudio } = useAudioStore();
  const [processing, setProcessing] = useState(false);
  const [moodData, setMoodData] = useState<any>(null);

  useEffect(() => {
    if (selectedAudio) {
      loadMoodData();
    }
  }, [selectedAudio]);

  const loadMoodData = async () => {
    if (!selectedAudio) return;

    try {
      const response = await moodAPI.get(selectedAudio.id);
      setMoodData(response.data.data);
    } catch (error) {
      // No mood data yet
    }
  };

  const handleClassify = async () => {
    if (!selectedAudio) {
      toast.error('Please select an audio file first');
      return;
    }

    setProcessing(true);

    try {
      await moodAPI.classify(selectedAudio.id);
      toast.success('Mood classification started');

      // Poll for results
      setTimeout(async () => {
        try {
          const response = await moodAPI.get(selectedAudio.id);
          setMoodData(response.data.data);
          setProcessing(false);
          toast.success('Mood classification completed');
        } catch (error) {
          setProcessing(false);
          toast.error('Classification still processing. Please check back later.');
        }
      }, 5000);
    } catch (error: any) {
      setProcessing(false);
      toast.error('Failed to classify mood');
    }
  };

  if (!selectedAudio) {
    return (
      <Card>
        <CardContent className="p-12 text-center">
          <Smile className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
          <p className="text-lg text-muted-foreground">
            Select an audio file to classify its mood
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6"
    >
      <Card>
        <CardHeader>
          <CardTitle>Mood Classification</CardTitle>
          <CardDescription>
            Analyze the emotional content of {selectedAudio.original_name}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <Button
            onClick={handleClassify}
            disabled={processing}
            className="w-full"
          >
            {processing ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Analyzing Mood...
              </>
            ) : (
              <>
                <Smile className="w-4 h-4 mr-2" />
                Classify Mood
              </>
            )}
          </Button>

          {moodData && (
            <>
              <div className="p-6 rounded-lg bg-gradient-to-br from-primary/10 to-secondary/10 border border-border">
                <div className="text-center mb-4">
                  <div className="text-6xl mb-2">
                    {moodEmojis[moodData.mood] || '🎵'}
                  </div>
                  <h3 className="text-2xl font-bold text-foreground capitalize">
                    {moodData.mood}
                  </h3>
                  <p className="text-sm text-muted-foreground mt-1">
                    Confidence: {(moodData.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              {moodData.mood_scores && (
                <div className="space-y-3">
                  <h4 className="text-sm font-medium text-foreground flex items-center gap-2">
                    <TrendingUp className="w-4 h-4" />
                    Mood Distribution
                  </h4>
                  {Object.entries(moodData.mood_scores)
                    .sort(([, a], [, b]) => (b as number) - (a as number))
                    .slice(0, 5)
                    .map(([mood, score]) => (
                      <div key={mood} className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="text-foreground capitalize flex items-center gap-2">
                            <span className="text-lg">
                              {moodEmojis[mood] || '🎵'}
                            </span>
                            {mood}
                          </span>
                          <span className="text-muted-foreground">
                            {((score as number) * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="relative h-2 bg-secondary rounded-full overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${(score as number) * 100}%` }}
                            transition={{ duration: 0.5 }}
                            className={`h-full ${moodColors[mood] || 'bg-primary'}`}
                          />
                        </div>
                      </div>
                    ))}
                </div>
              )}

              {moodData.features && (
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 rounded-lg bg-secondary/50">
                    <div className="text-2xl font-bold text-foreground">
                      {moodData.features.tempo?.toFixed(0) || 'N/A'}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      BPM (Tempo)
                    </div>
                  </div>
                  <div className="p-4 rounded-lg bg-secondary/50">
                    <div className="text-2xl font-bold text-foreground">
                      {moodData.features.energy?.toFixed(2) || 'N/A'}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Energy Level
                    </div>
                  </div>
                  <div className="p-4 rounded-lg bg-secondary/50">
                    <div className="text-2xl font-bold text-foreground">
                      {moodData.features.valence?.toFixed(2) || 'N/A'}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Valence
                    </div>
                  </div>
                  <div className="p-4 rounded-lg bg-secondary/50">
                    <div className="text-2xl font-bold text-foreground">
                      {moodData.features.danceability?.toFixed(2) || 'N/A'}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Danceability
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}
