import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface User {
  id: number;
  email: string;
  username: string;
  created_at: string;
}

interface AudioFile {
  id: number;
  filename: string;
  original_name: string;
  file_size: number;
  duration: number;
  format: string;
  created_at: string;
}

interface ProcessingJob {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  type: string;
}

interface AuthStore {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (user: User, token: string) => void;
  logout: () => void;
  setUser: (user: User) => void;
}

export const useAuthStore = create<AuthStore>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      login: (user, token) => {
        localStorage.setItem('token', token);
        set({ user, token, isAuthenticated: true });
      },
      logout: () => {
        localStorage.removeItem('token');
        set({ user: null, token: null, isAuthenticated: false });
      },
      setUser: (user) => set({ user }),
    }),
    {
      name: 'auth-storage',
    }
  )
);

interface AudioStore {
  audioFiles: AudioFile[];
  selectedAudio: AudioFile | null;
  isLoading: boolean;
  setAudioFiles: (files: AudioFile[]) => void;
  addAudioFile: (file: AudioFile) => void;
  removeAudioFile: (id: number) => void;
  selectAudio: (audio: AudioFile | null) => void;
  setLoading: (loading: boolean) => void;
}

export const useAudioStore = create<AudioStore>((set) => ({
  audioFiles: [],
  selectedAudio: null,
  isLoading: false,
  setAudioFiles: (files) => set({ audioFiles: files }),
  addAudioFile: (file) => set((state) => ({
    audioFiles: [file, ...state.audioFiles]
  })),
  removeAudioFile: (id) => set((state) => ({
    audioFiles: state.audioFiles.filter(f => f.id !== id),
    selectedAudio: state.selectedAudio?.id === id ? null : state.selectedAudio
  })),
  selectAudio: (audio) => set({ selectedAudio: audio }),
  setLoading: (loading) => set({ isLoading: loading }),
}));

interface JobStore {
  jobs: Map<string, ProcessingJob>;
  addJob: (job: ProcessingJob) => void;
  updateJob: (id: string, updates: Partial<ProcessingJob>) => void;
  removeJob: (id: string) => void;
  getJob: (id: string) => ProcessingJob | undefined;
}

export const useJobStore = create<JobStore>((set, get) => ({
  jobs: new Map(),
  addJob: (job) => set((state) => {
    const newJobs = new Map(state.jobs);
    newJobs.set(job.id, job);
    return { jobs: newJobs };
  }),
  updateJob: (id, updates) => set((state) => {
    const newJobs = new Map(state.jobs);
    const job = newJobs.get(id);
    if (job) {
      newJobs.set(id, { ...job, ...updates });
    }
    return { jobs: newJobs };
  }),
  removeJob: (id) => set((state) => {
    const newJobs = new Map(state.jobs);
    newJobs.delete(id);
    return { jobs: newJobs };
  }),
  getJob: (id) => get().jobs.get(id),
}));

interface UIStore {
  sidebarOpen: boolean;
  currentTab: string;
  theme: 'light' | 'dark';
  toggleSidebar: () => void;
  setCurrentTab: (tab: string) => void;
  setTheme: (theme: 'light' | 'dark') => void;
}

export const useUIStore = create<UIStore>()(
  persist(
    (set) => ({
      sidebarOpen: true,
      currentTab: 'upload',
      theme: 'dark',
      toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
      setCurrentTab: (tab) => set({ currentTab: tab }),
      setTheme: (theme) => set({ theme }),
    }),
    {
      name: 'ui-storage',
    }
  )
);
