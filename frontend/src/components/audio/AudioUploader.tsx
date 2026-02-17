'use client';

import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Music, CheckCircle, XCircle, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';
import { audioAPI } from '@/lib/api';
import { useAudioStore } from '@/lib/store';
import { Card } from '@/components/ui/card';
import { formatBytes } from '@/lib/utils';
import toast from 'react-hot-toast';

export function AudioUploader() {
  const [uploading, setUploading] = useState(false);
  const addAudioFile = useAudioStore(state => state.addAudioFile);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setUploading(true);

    const loadingToast = toast.loading(`Uploading ${file.name}...`);

    try {
      const response = await audioAPI.upload(file);
      addAudioFile(response.data.data.audioFile);
      
      toast.success('Audio uploaded successfully', {
        id: loadingToast,
      });
    } catch (err: any) {
      toast.error(err.response?.data?.message || 'Upload failed', {
        id: loadingToast,
      });
    } finally {
      setUploading(false);
    }
  }, [addAudioFile]);

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac']
    },
    maxFiles: 1,
    maxSize: 100 * 1024 * 1024,
    disabled: uploading
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full"
    >
      <Card
        {...getRootProps()}
        className={`
          relative border-2 border-dashed p-12 text-center cursor-pointer
          transition-all duration-300
          ${isDragActive 
            ? 'border-primary bg-primary/10' 
            : 'border-border hover:border-primary/50 hover:bg-accent/50'
          }
          ${uploading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        <div className="flex flex-col items-center gap-4">
          {uploading ? (
            <>
              <Loader2 className="w-16 h-16 text-primary animate-spin" />
              <p className="text-lg text-foreground">Uploading...</p>
            </>
          ) : (
            <>
              <div className="p-4 rounded-full bg-primary/10">
                <Upload className="w-12 h-12 text-primary" />
              </div>
              <div>
                <p className="text-xl font-semibold text-foreground mb-2">
                  Drop your audio file here
                </p>
                <p className="text-sm text-muted-foreground">
                  or click to browse
                </p>
                <p className="text-xs text-muted-foreground mt-2">
                  Supported formats: MP3, WAV, FLAC, OGG, M4A, AAC
                </p>
                <p className="text-xs text-muted-foreground">
                  Maximum file size: 100MB
                </p>
              </div>
            </>
          )}
        </div>

        {fileRejections.length > 0 && (
          <div className="mt-4 p-3 bg-destructive/10 border border-destructive rounded-md">
            <p className="text-sm text-destructive">
              {fileRejections[0].errors[0].message}
            </p>
          </div>
        )}
      </Card>
    </motion.div>
  );
}
