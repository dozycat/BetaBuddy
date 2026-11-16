// Keep the HarmonyOS import limits aligned with iOS 1.4 and Android.
export const MAX_VIDEO_BYTES: number = 500 * 1024 * 1024;
export const MIN_VIDEO_SECONDS: number = 0.5;
export const MAX_VIDEO_SECONDS: number = 20 * 60;

export enum ImportError {
  INVALID_VIDEO = 'invalidVideo',
  TOO_LONG = 'tooLong',
  TOO_LARGE = 'tooLarge',
}

export function checkImport(sizeBytes: number, durationSeconds: number | null,
  hasVideoTrack: boolean): ImportError | null {
  if (sizeBytes > MAX_VIDEO_BYTES) return ImportError.TOO_LARGE;
  if (sizeBytes <= 0 || !hasVideoTrack || durationSeconds === null ||
    !Number.isFinite(durationSeconds) || durationSeconds < MIN_VIDEO_SECONDS) return ImportError.INVALID_VIDEO;
  if (durationSeconds > MAX_VIDEO_SECONDS) return ImportError.TOO_LONG;
  return null;
}
