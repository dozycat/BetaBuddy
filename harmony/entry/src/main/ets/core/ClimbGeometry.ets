// Platform-independent geometry used by the HarmonyOS port. Coordinates are normalized from
// the top left, matching ios/BetaBuddy/Core/Models.swift and Android's data/Models.kt.
export interface Joint {
  x: number;
  y: number;
  confidence: number;
}

export interface PoseFrame {
  time: number;
  joints: Joint[];
}

export interface Connection {
  from: number;
  to: number;
}

export const CONNECTIONS: Connection[] = [
  { from: 11, to: 12 }, { from: 11, to: 13 }, { from: 13, to: 15 },
  { from: 12, to: 14 }, { from: 14, to: 16 }, { from: 11, to: 23 },
  { from: 12, to: 24 }, { from: 23, to: 24 }, { from: 23, to: 25 },
  { from: 25, to: 27 }, { from: 24, to: 26 }, { from: 26, to: 28 },
  { from: 27, to: 29 }, { from: 29, to: 31 }, { from: 27, to: 31 },
  { from: 28, to: 30 }, { from: 30, to: 32 }, { from: 28, to: 32 },
];

export function validJoint(joint: Joint): boolean {
  return Number.isFinite(joint.x) && Number.isFinite(joint.y) &&
    joint.confidence >= 0.6 && joint.x >= 0 && joint.x <= 1 &&
    joint.y >= 0 && joint.y <= 1;
}

export function hip(frame: PoseFrame): Joint | null {
  if (frame.joints.length !== 33 || !validJoint(frame.joints[23]) || !validJoint(frame.joints[24])) {
    return null;
  }
  const left = frame.joints[23];
  const right = frame.joints[24];
  return {
    x: (left.x + right.x) / 2,
    y: (left.y + right.y) / 2,
    confidence: Math.min(left.confidence, right.confidence),
  };
}

export function hasReliableTorso(frame: PoseFrame): boolean {
  return frame.joints.length === 33 && validJoint(frame.joints[11]) &&
    validJoint(frame.joints[12]) && validJoint(frame.joints[23]) &&
    validJoint(frame.joints[24]);
}

// This is a projected 2D angle in the picture, not a physical joint angle.
export function angle(frame: PoseFrame, a: number, b: number, c: number, aspect: number): number | null {
  if (!Number.isFinite(aspect) || aspect <= 0) return null;
  if (a < 0 || b < 0 || c < 0 ||
    a >= frame.joints.length || b >= frame.joints.length || c >= frame.joints.length) return null;
  const p = frame.joints[a];
  const q = frame.joints[b];
  const r = frame.joints[c];
  if (!validJoint(p) || !validJoint(q) || !validJoint(r)) return null;
  const ux = (p.x - q.x) * aspect;
  const uy = p.y - q.y;
  const vx = (r.x - q.x) * aspect;
  const vy = r.y - q.y;
  const length = Math.hypot(ux, uy) * Math.hypot(vx, vy);
  if (length <= 0.000001) return null;
  const cosine = Math.max(-1, Math.min(1, (ux * vx + uy * vy) / length));
  return Math.acos(cosine) * 180 / Math.PI;
}

export function coverage(frames: PoseFrame[], sampledFrames: number): number {
  if (sampledFrames <= 0) return 0;
  let reliable = 0;
  for (const frame of frames) {
    if (hip(frame) !== null) reliable++;
  }
  return reliable / sampledFrames;
}

export function averageElbowAngle(frames: PoseFrame[], aspect: number): number | null {
  let sum = 0;
  let count = 0;
  for (const frame of frames) {
    const left = angle(frame, 11, 13, 15, aspect);
    const right = angle(frame, 12, 14, 16, aspect);
    if (left !== null) { sum += left; count++; }
    if (right !== null) { sum += right; count++; }
  }
  return count === 0 ? null : sum / count;
}

// Frames are stored in time order. Never draw a stale skeleton across a missing detection.
export function nearestFrame(frames: PoseFrame[], time: number): PoseFrame | null {
  let low = 0;
  let high = frames.length;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (frames[mid].time < time) low = mid + 1;
    else high = mid;
  }
  let best: PoseFrame | null = null;
  if (low > 0) best = frames[low - 1];
  if (low < frames.length &&
    (best === null || Math.abs(frames[low].time - time) < Math.abs(best.time - time))) {
    best = frames[low];
  }
  return best !== null && Math.abs(best.time - time) <= 0.16 && hasReliableTorso(best) ? best : null;
}
