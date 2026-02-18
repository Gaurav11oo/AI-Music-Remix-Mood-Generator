"use client";

import { useEffect, useState } from "react";
import {
  Music,
  Download,
  Trash2,
  Play,
  Pause,
  MoreVertical,
} from "lucide-react";
import { motion } from "framer-motion";
import { audioAPI } from "@/lib/api";
import { useAudioStore } from "@/lib/store";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { formatBytes, formatDuration } from "@/lib/utils";
import toast from "react-hot-toast";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export function AudioList() {
  const {
    audioFiles,
    setAudioFiles,
    removeAudioFile,
    selectAudio,
    selectedAudio,
  } = useAudioStore();
  const [loading, setLoading] = useState(true);
  const [playingId, setPlayingId] = useState<number | null>(null);

  useEffect(() => {
    loadAudioFiles();
  }, []);

  const loadAudioFiles = async () => {
    try {
      const response = await audioAPI.list();
      setAudioFiles(response.data.data.audioFiles);
    } catch (error: any) {
      toast.error("Failed to load audio files");
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: number, e: React.MouseEvent) => {
    e.stopPropagation();

    if (!confirm("Are you sure you want to delete this audio file?")) {
      return;
    }

    try {
      await audioAPI.delete(id);
      removeAudioFile(id);
      toast.success("Audio file deleted");
    } catch (error: any) {
      toast.error("Failed to delete audio file");
    }
  };

  const handleDownload = async (
    id: number,
    filename: string,
    e: React.MouseEvent
  ) => {
    e.stopPropagation();

    try {
      const response = await audioAPI.download(id);
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      toast.success("Download started");
    } catch (error: any) {
      toast.error("Failed to download audio file");
    }
  };

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {[1, 2, 3].map((i) => (
          <Card key={i} className="animate-pulse">
            <CardContent className="p-6">
              <div className="h-20 bg-secondary rounded" />
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  if (audioFiles.length === 0) {
    return (
      <Card className="p-12 text-center">
        <Music className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
        <p className="text-lg text-muted-foreground">
          No audio files yet. Upload your first track to get started.
        </p>
      </Card>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {audioFiles.map((audio, index) => (
        <motion.div
          key={audio.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.05 }}
        >
          <Card
            className={`cursor-pointer transition-all hover:shadow-lg ${
              selectedAudio?.id === audio.id ? "ring-2 ring-primary" : ""
            }`}
            onClick={() => selectAudio(audio)}
          >
            <CardContent className="p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3 flex-1 min-w-0">
                  <div className="p-3 rounded-lg bg-primary/10">
                    <Music className="w-6 h-6 text-primary" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold text-foreground truncate">
                      {audio.original_name}
                    </h3>
                    <p className="text-sm text-muted-foreground">
                      {audio.format?.toUpperCase()} •{" "}
                      {formatBytes(audio.file_size)}
                    </p>
                  </div>
                </div>

                <DropdownMenu>
                  <DropdownMenuTrigger
                    asChild
                    onClick={(e) => e.stopPropagation()}
                  >
                    <Button variant="ghost" size="icon">
                      <MoreVertical className="w-4 h-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem
                      onClick={(e) =>
                        handleDownload(audio.id, audio.original_name, e as any)
                      }
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Download
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onClick={(e) => handleDelete(audio.id, e as any)}
                      className="text-destructive"
                    >
                      <Trash2 className="w-4 h-4 mr-2" />
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Duration</span>
                  <span className="text-foreground font-medium">
                    {audio.duration ? formatDuration(audio.duration) : "N/A"}
                  </span>
                </div>
                {audio.sample_rate && (
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Sample Rate</span>
                    <span className="text-foreground font-medium">
                      {audio.sample_rate} Hz
                    </span>
                  </div>
                )}
                {audio.channels && (
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Channels</span>
                    <span className="text-foreground font-medium">
                      {audio.channels === 1 ? "Mono" : "Stereo"}
                    </span>
                  </div>
                )}
              </div>

              <div className="mt-4 pt-4 border-t border-border">
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full"
                  onClick={(e) => {
                    e.stopPropagation();
                    selectAudio(audio);
                  }}
                >
                  Select for Processing
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      ))}
    </div>
  );
}
