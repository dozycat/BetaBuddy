export interface Video {
  id: string;
  filename: string;
  original_filename: string;
  file_path: string;
  file_size: number;
  duration: number | null;
  fps: number | null;
  width: number | null;
  height: number | null;
  total_frames: number | null;
  created_at: string;
  preview_url: string | null;
}

export interface VideoListResponse {
  videos: Video[];
  total: number;
}

export interface AnalysisTask {
  id: string;
  video_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  websocket_url: string | null;
}

export interface JointAngleStats {
  min: number;
  max: number;
  avg: number;
}

export interface SummaryStats {
  duration: number;
  avg_stability_score: number;
  min_stability_score: number;
  max_stability_score: number;
  avg_efficiency: number;
  max_velocity: number;
  max_acceleration: number;
  dyno_count: number;
  total_distance: number;
}

export interface AnalysisResult {
  id: string;
  task_id: string;
  total_frames_analyzed: number;
  avg_stability_score: number | null;
  avg_efficiency: number | null;
  max_acceleration: number | null;
  dyno_detected: number;
  summary_stats: SummaryStats | null;
  joint_angle_stats: Record<string, JointAngleStats> | null;
  com_trajectory: [number, number][] | null;
  beta_suggestion: string | null;
  annotated_video_url: string | null;
  created_at: string;
}

export interface AnnotateResponse {
  video_id: string;
  annotated_video_url: string;
  message: string;
}

export interface BetaSuggestion {
  video_id: string;
  suggestion: string;
  metrics_used: Record<string, unknown>;
  generated_at: string;
}

export interface WSMessage {
  type: 'progress' | 'keypoints' | 'metrics' | 'complete' | 'error' | 'status' | 'pong';
  data?: {
    progress?: number;
    current_frame?: number;
    status?: string;
    task_id?: string;
    result_id?: string;
    // Keypoints message data
    frame_number?: number;
    keypoints?: Array<{
      x: number;
      y: number;
      z?: number;
      visibility?: number;
      name?: string;
    }>;
    center_of_mass?: [number, number];
    // Metrics message data
    joint_angles?: Record<string, number>;
    stability_score?: number;
    velocity?: [number, number];
    acceleration?: [number, number];
    // Complete message data
    summary?: SummaryStats;
    // Error message data
    message?: string;
  };
  message?: string;
}
