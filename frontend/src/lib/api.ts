import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

// Create axios instance
const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Auth API
export const authAPI = {
  register: (data: { email: string; username: string; password: string }) =>
    apiClient.post('/auth/register', data),
  
  login: (data: { email: string; password: string }) =>
    apiClient.post('/auth/login', data),
  
  getCurrentUser: () =>
    apiClient.get('/auth/me'),
};

// Audio API
export const audioAPI = {
  upload: (file: File) => {
    const formData = new FormData();
    formData.append('audio', file);
    return apiClient.post('/audio/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  
  list: (params?: { page?: number; limit?: number }) =>
    apiClient.get('/audio', { params }),
  
  get: (id: number) =>
    apiClient.get(`/audio/${id}`),
  
  delete: (id: number) =>
    apiClient.delete(`/audio/${id}`),
  
  download: (id: number) =>
    apiClient.get(`/audio/${id}/download`, { responseType: 'blob' }),
  
  getWaveform: (id: number, samples?: number) =>
    apiClient.get(`/audio/${id}/waveform`, { params: { samples } }),
  
  getSpectrogram: (id: number) =>
    apiClient.get(`/audio/${id}/spectrogram`, { responseType: 'blob' }),
  
  getFeatures: (id: number) =>
    apiClient.get(`/audio/${id}/features`),
};

// Stems API
export const stemsAPI = {
  separate: (data: { audioFileId: number; model?: string; stems?: string[] }) =>
    apiClient.post('/stems/separate', data),
  
  getStatus: (jobId: string) =>
    apiClient.get(`/stems/${jobId}/status`),
  
  download: (jobId: string) =>
    apiClient.get(`/stems/${jobId}/download`),
};

// Mood API
export const moodAPI = {
  classify: (audioFileId: number) =>
    apiClient.post('/mood/classify', { audioFileId }),
  
  get: (audioId: number) =>
    apiClient.get(`/mood/${audioId}`),
};

// Generate API
export const generateAPI = {
  textToMusic: (data: {
    prompt: string;
    duration?: number;
    temperature?: number;
    top_k?: number;
  }) =>
    apiClient.post('/generate/text-to-music', data),
  
  getStatus: (jobId: string) =>
    apiClient.get(`/generate/${jobId}/status`),
  
  download: (jobId: string) =>
    apiClient.get(`/generate/${jobId}/download`, { responseType: 'blob' }),
};

// Remix API
export const remixAPI = {
  genre: (data: { audioFileId: number; targetGenre: string; intensity?: number }) =>
    apiClient.post('/remix/genre', data),
  
  tempo: (data: { audioFileId: number; tempoChange: number; preservePitch?: boolean }) =>
    apiClient.post('/remix/tempo', data),
  
  pitch: (data: { audioFileId: number; pitchChange: number }) =>
    apiClient.post('/remix/pitch', data),
  
  getStatus: (jobId: string) =>
    apiClient.get(`/remix/${jobId}/status`),
};

export default apiClient;
