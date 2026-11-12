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
  thumbnail_url: string | null;
}

export interface ThumbnailResponse {
  video_id: string;
  thumbnail_url: string;
  message: string;
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
  avg_efficiency: number;
  max_velocity: number;
  max_acceleration: number;
  dyno_count: number;
  total_distance: number;
  meters_per_unit?: number;  // Conversion factor from normalized units to meters
}

export interface AnalysisResult {
  id: string;
  task_id: string;
  total_frames_analyzed: number;
  avg_efficiency: number | null;
  max_acceleration: number | null;
  dyno_detected: number;
  summary_stats: SummaryStats | null;
  joint_angle_stats: Record<string, JointAngleStats> | null;
  com_trajectory: [number, number][] | null;
  beta_suggestion: string | null;
  annotated_video_url: string | null;
  meters_per_unit: number | null;  // Conversion factor from normalized units to meters
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
    velocity?: [number, number];
    acceleration?: [number, number];
    // Complete message data
    summary?: SummaryStats;
    // Error message data
    message?: string;
  };
  message?: string;
}

// Movement detection types
export interface DetectedMovement {
  movement_type: string;
  movement_name_cn: string;
  start_frame: number;
  end_frame: number;
  start_timestamp: number;
  end_timestamp: number;
  side: 'left' | 'right' | 'both';
  side_cn: string;
  confidence: number;
  is_challenging: boolean;
  key_angles: Record<string, number>;
  peak_frame: number;
  description_cn?: string;
}

export interface TimelineEntry {
  timestamp: number;
  frame: number;
  movements: string[];
}

export interface MovementSummary {
  total_movements: number;
  by_type: Record<string, number>;
  challenging_count: number;
  total_duration: number;
}

export interface MovementDetectionResponse {
  video_id: string;
  total_movements: number;
  challenging_count: number;
  movements: DetectedMovement[];
  timeline: TimelineEntry[];
  summary: MovementSummary;
}

export interface MovementDetectionRequest {
  min_duration_frames?: number;
  confidence_threshold?: number;
  generate_descriptions?: boolean;
}
