export interface AudioFile {
  id: number;
  original_name: string;
  format?: string;
  file_size: number;
  duration?: number;

  // ✅ REQUIRED
  sample_rate?: number;
  channels?: number;
}
