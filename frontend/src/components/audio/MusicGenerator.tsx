'use client';

import { useState } from 'react';
import { Wand2, Download, Loader2, Play } from 'lucide-react';
import { motion } from 'framer-motion';
import { generateAPI } from '@/lib/api';
import { useJobStore } from '@/lib/store';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import { Progress } from '@/components/ui/progress';
import toast from 'react-hot-toast';

export function MusicGenerator() {
  const { addJob, updateJob } = useJobStore();
  const [prompt, setPrompt] = useState('');
  const [duration, setDuration] = useState([10]);
  const [temperature, setTemperature] = useState([1.0]);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [generatedAudio, setGeneratedAudio] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      toast.error('Please enter a prompt');
      return;
    }

    setProcessing(true);
    setProgress(0);
    setGeneratedAudio(null);

    try {
      const response = await generateAPI.textToMusic({
        prompt: prompt.trim(),
        duration: duration[0],
        temperature: temperature[0]
      });

      const jobId = response.data.data.jobId;
      setCurrentJobId(jobId);

      addJob({
        id: jobId,
        status: 'pending',
        progress: 0,
        type: 'music_generation'
      });

      pollJobStatus(jobId);
      toast.success('Music generation started');
    } catch (error: any) {
      toast.error('Failed to start music generation');
      setProcessing(false);
    }
  };

  const pollJobStatus = async (jobId: string) => {
    const interval = setInterval(async () => {
      try {
        const response = await generateAPI.getStatus(jobId);
        const job = response.data.data;

        setProgress(job.progress || 0);
        updateJob(jobId, {
          status: job.status,
          progress: job.progress
        });

        if (job.status === 'completed') {
          clearInterval(interval);
          setProcessing(false);
          setGeneratedAudio(job.result.audio_path);
          toast.success('Music generation completed');
        } else if (job.status === 'failed') {
          clearInterval(interval);
          setProcessing(false);
          toast.error(job.error || 'Music generation failed');
        }
      } catch (error) {
        clearInterval(interval);
        setProcessing(false);
        toast.error('Failed to check job status');
      }
    }, 3000);
  };

  const handleDownload = async () => {
    if (!currentJobId) return;

    try {
      const response = await generateAPI.download(currentJobId);
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'generated-music.wav');
      document.body.appendChild(link);
      link.click();
      link.remove();
      toast.success('Download started');
    } catch (error: any) {
      toast.error('Failed to download music');
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6"
    >
      <Card>
        <CardHeader>
          <CardTitle>Text-to-Music Generator</CardTitle>
          <CardDescription>
            Generate original music from text descriptions using AI
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">
              Music Description
            </label>
            <Input
              placeholder="e.g., upbeat electronic dance music with energetic drums"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={processing}
            />
            <p className="text-xs text-muted-foreground">
              Describe the style, mood, instruments, and characteristics you want
            </p>
          </div>

          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between">
                <label className="text-sm font-medium text-foreground">
                  Duration
                </label>
                <span className="text-sm text-muted-foreground">
                  {duration[0]} seconds
                </span>
              </div>
              <Slider
                value={duration}
                onValueChange={setDuration}
                min={5}
                max={30}
                step={1}
                disabled={processing}
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <label className="text-sm font-medium text-foreground">
                  Creativity
                </label>
                <span className="text-sm text-muted-foreground">
                  {temperature[0].toFixed(1)}
                </span>
              </div>
              <Slider
                value={temperature}
                onValueChange={setTemperature}
                min={0.5}
                max={1.5}
                step={0.1}
                disabled={processing}
              />
              <p className="text-xs text-muted-foreground">
                Higher values create more varied and experimental music
              </p>
            </div>
          </div>

          {processing && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Generating</span>
                <span className="text-foreground font-medium">{progress}%</span>
              </div>
              <Progress value={progress} className="w-full" />
              <p className="text-xs text-muted-foreground text-center">
                This may take a few minutes...
              </p>
            </div>
          )}

          <Button
            onClick={handleGenerate}
            disabled={processing || !prompt.trim()}
            className="w-full"
          >
            {processing ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Generating Music...
              </>
            ) : (
              <>
                <Wand2 className="w-4 h-4 mr-2" />
                Generate Music
              </>
            )}
          </Button>

          {generatedAudio && (
            <div className="p-4 rounded-lg bg-secondary/50 space-y-3">
              <p className="text-sm font-medium text-foreground">
                Music Generated Successfully
              </p>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  onClick={handleDownload}
                  className="flex-1"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Download
                </Button>
              </div>
            </div>
          )}

          <div className="p-4 rounded-lg bg-primary/10 border border-primary/20">
            <h4 className="text-sm font-medium text-foreground mb-2">
              Example Prompts
            </h4>
            <ul className="space-y-1 text-xs text-muted-foreground">
              <li>• Upbeat jazz piano with walking bass and light drums</li>
              <li>• Ambient electronic soundscape with soft synth pads</li>
              <li>• Energetic rock guitar riff with driving drums</li>
              <li>• Peaceful acoustic guitar melody with nature sounds</li>
              <li>• Dark cinematic orchestral music with deep strings</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
