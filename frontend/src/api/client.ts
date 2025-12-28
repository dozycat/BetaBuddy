import axios from 'axios';
import type { Video, VideoListResponse, AnalysisTask, AnalysisResult, BetaSuggestion } from '../types';

const api = axios.create({
  baseURL: '/api/v1',
});

export const videoApi = {
  upload: async (file: File): Promise<Video> => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post<Video>('/videos/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  list: async (skip = 0, limit = 20): Promise<VideoListResponse> => {
    const response = await api.get<VideoListResponse>('/videos', {
      params: { skip, limit },
    });
    return response.data;
  },

  get: async (videoId: string): Promise<Video> => {
    const response = await api.get<Video>(`/videos/${videoId}`);
    return response.data;
  },

  delete: async (videoId: string): Promise<void> => {
    await api.delete(`/videos/${videoId}`);
  },
};

export const analysisApi = {
  start: async (videoId: string): Promise<AnalysisTask> => {
    const response = await api.get<AnalysisTask>(`/videos/${videoId}/analyze`);
    return response.data;
  },

  getResults: async (videoId: string): Promise<AnalysisResult> => {
    const response = await api.get<AnalysisResult>(`/videos/${videoId}/results`);
    return response.data;
  },

  getTaskStatus: async (taskId: string): Promise<AnalysisTask> => {
    const response = await api.get<AnalysisTask>(`/tasks/${taskId}`);
    return response.data;
  },
};

export const betaApi = {
  getSuggestion: async (videoId: string): Promise<BetaSuggestion> => {
    const response = await api.post<BetaSuggestion>('/beta/suggest', {
      video_id: videoId,
    });
    return response.data;
  },

  checkStatus: async (): Promise<{ available: boolean; model: string }> => {
    const response = await api.get('/beta/status');
    return response.data;
  },
};

export default api;
