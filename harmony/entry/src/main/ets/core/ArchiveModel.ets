import { PoseFrame } from './ClimbGeometry';

// Keep field names aligned with the iOS and Android JSON archive. Dates use seconds since
// 2001-01-01T00:00:00Z, as Foundation's JSONEncoder writes them.
export const REFERENCE_DATE_OFFSET_SECONDS: number = 978307200;
export const UNGRADED_VALUE: string = '未标级';

export interface Moment {
  id: string;
  time: number;
  title: string;
  detail: string;
  kind?: string;
}

export interface PoseAnalysis {
  frames: PoseFrame[];
  sampledFrames: number;
  duration: number;
  aspectRatio: number;
  processingSeconds: number;
  moments: Moment[];
}

export interface Climb {
  id: string;
  date: number;
  title: string;
  grade: string;
  sent: boolean;
  note: string;
  videoName: string;
  thumbnailName: string;
  analysis: PoseAnalysis;
  gymId?: string;
  gymName?: string;
  climbedAt?: number;
  folderId?: string;
  pendingAnalysis?: string;
}

export interface ClimbFolder {
  id: string;
  name: string;
}

export interface Archive {
  version: number;
  climbs: Climb[];
  folders: ClimbFolder[];
}

export function climbingDate(climb: Climb): number {
  return climb.climbedAt === undefined ? climb.date : climb.climbedAt;
}

export function gymKey(climb: Climb): string {
  if (climb.gymId !== undefined && climb.gymId.length > 0) return 'id:' + climb.gymId;
  const name = (climb.gymName === undefined ? '' : climb.gymName).trim();
  return name.length === 0 ? '' : 'name:' + name;
}

export function unixMillis(date: number): number {
  return Math.trunc((date + REFERENCE_DATE_OFFSET_SECONDS) * 1000);
}

export function archiveDate(unixMillisValue: number): number {
  return unixMillisValue / 1000 - REFERENCE_DATE_OFFSET_SECONDS;
}
