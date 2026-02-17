'use client';

import { useState } from 'react';
import { Scissors, Download, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';
import { stemsAPI } from '@/lib/api';
import { useAudioStore, useJobStore } from '@/lib/store';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import toast from 'react-hot-toast';

export function StemSeparation() {
  const { selectedAudio } = useAudioStore();
  const { addJob, updateJob, getJob } = useJobStore();
  const [processing, setProcessing] = useState(false);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [stems, setStems] = useState<any>(null);

  const handleSeparate = async () => {
    if (!selectedAudio) {
      toast.error('Please select an audio file first');
      return;
    }

    setProcessing(true);
    setProgress(0);

    try {
      const response = await stemsAPI.separate({
        audioFileId: selectedAudio.id,
        model: 'htdemucs',
        stems: ['vocals', 'drums', 'bass', 'other']
      });

      const jobId = response.data.data.jobId;
      setCurrentJobId(jobId);

      addJob({
        id: jobId,
        status: 'pending',
        progress: 0,
        type: 'stem_separation'
      });

      pollJobStatus(jobId);
    } catch (error: any) {
      toast.error('Failed to start stem separation');
      setProcessing(false);
    }
  };

  const pollJobStatus = async (jobId: string) => {
    const interval = setInterval(async () => {
      try {
        const response = await stemsAPI.getStatus(jobId);
        const job = response.data.data;

        setProgress(job.progress || 0);
        updateJob(jobId, {
          status: job.status,
          progress: job.progress
        });

        if (job.status === 'completed') {
          clearInterval(interval);
          setProcessing(false);
          setStems(job.result);
          toast.success('Stem separation completed');
        } else if (job.status === 'failed') {
          clearInterval(interval);
          setProcessing(false);
          toast.error(job.error || 'Stem separation failed');
        }
      } catch (error) {
        clearInterval(interval);
        setProcessing(false);
        toast.error('Failed to check job status');
      }
    }, 2000);
  };

  const handleDownloadStems = async () => {
    if (!currentJobId) return;

    try {
      const response = await stemsAPI.download(currentJobId);
      toast.success('Downloading stems');
    } catch (error: any) {
      toast.error('Failed to download stems');
    }
  };

  if (!selectedAudio) {
    return (
      <Card>
        <CardContent className="p-12 text-center">
          <Scissors className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
          <p className="text-lg text-muted-foreground">
            Select an audio file to separate stems
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
          <CardTitle>Stem Separation</CardTitle>
          <CardDescription>
            Separate {selectedAudio.original_name} into individual stems
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {['Vocals', 'Drums', 'Bass', 'Other'].map((stem) => (
              <div
                key={stem}
                className="p-4 rounded-lg border border-border bg-card text-center"
              >
                <div className="text-sm font-medium text-foreground">{stem}</div>
                <div className="text-xs text-muted-foreground mt-1">
                  {processing ? 'Processing...' : 'Ready'}
                </div>
              </div>
            ))}
          </div>

          {processing && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Progress</span>
                <span className="text-foreground font-medium">{progress}%</span>
              </div>
              <Progress value={progress} className="w-full" />
            </div>
          )}

          <div className="flex gap-4">
            <Button
              onClick={handleSeparate}
              disabled={processing}
              className="flex-1"
            >
              {processing ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Separating Stems...
                </>
              ) : (
                <>
                  <Scissors className="w-4 h-4 mr-2" />
                  Separate Stems
                </>
              )}
            </Button>

            {stems && (
              <Button
                variant="outline"
                onClick={handleDownloadStems}
              >
                <Download className="w-4 h-4 mr-2" />
                Download Stems
              </Button>
            )}
          </div>

          {stems && (
            <div className="p-4 rounded-lg bg-secondary/50">
              <p className="text-sm font-medium text-foreground mb-2">
                Separation Complete
              </p>
              <div className="space-y-1">
                {Object.entries(stems.stems || {}).map(([stem, path]) => (
                  <div key={stem} className="flex justify-between text-sm">
                    <span className="text-muted-foreground capitalize">{stem}</span>
                    <span className="text-foreground font-mono text-xs">
                      {typeof path === 'string' ? path.split('/').pop() : 'Ready'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}
