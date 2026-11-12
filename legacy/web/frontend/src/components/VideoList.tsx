import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { Video } from '../types';
import { videoApi } from '../api/client';

interface VideoListProps {
  videos: Video[];
  onSelect: (video: Video) => void;
  onDelete?: (videoId: string) => void;
  onVideoUpdate?: (video: Video) => void;
  selectedId?: string;
}

export const VideoList: React.FC<VideoListProps> = ({
  videos,
  onSelect,
  onDelete,
  onVideoUpdate,
  selectedId,
}) => {
  const { t } = useTranslation();
  const [regenerating, setRegenerating] = useState<string | null>(null);

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  };

  const formatDuration = (seconds: number | null) => {
    if (!seconds) return '--:--';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleRegenerateThumbnail = async (e: React.MouseEvent, video: Video) => {
    e.stopPropagation();
    setRegenerating(video.id);
    try {
      const result = await videoApi.regenerateThumbnail(video.id);
      if (onVideoUpdate) {
        onVideoUpdate({ ...video, thumbnail_url: result.thumbnail_url });
      }
    } catch (error) {
      console.error('Failed to regenerate thumbnail:', error);
    } finally {
      setRegenerating(null);
    }
  };

  if (videos.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        <p>{t('videoList.noVideos')}</p>
        <p className="text-sm mt-1">{t('videoList.getStarted')}</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {videos.map((video) => (
        <div
          key={video.id}
          className={`p-3 rounded-lg border cursor-pointer transition-all ${
            selectedId === video.id
              ? 'border-primary-500 bg-primary-50'
              : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
          }`}
          onClick={() => onSelect(video)}
        >
          <div className="flex items-start gap-3">
            {/* Thumbnail */}
            <div className="flex-shrink-0 w-32 h-20 bg-gray-100 rounded overflow-hidden">
              {video.thumbnail_url ? (
                <img
                  src={video.thumbnail_url}
                  alt={video.original_filename}
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-gray-400">
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                    />
                  </svg>
                </div>
              )}
            </div>

            {/* Video info */}
            <div className="flex-1 min-w-0">
              <h4 className="font-medium text-gray-900 truncate">
                {video.original_filename}
              </h4>
              <div className="flex items-center gap-2 mt-1 text-sm text-gray-500">
                <span>{formatDuration(video.duration)}</span>
                <span>•</span>
                <span>{formatFileSize(video.file_size)}</span>
                {video.width && video.height && (
                  <>
                    <span>•</span>
                    <span>{video.width}x{video.height}</span>
                  </>
                )}
              </div>
            </div>

            {/* Actions */}
            <div className="flex items-center gap-1">
              {/* Regenerate thumbnail button */}
              <button
                onClick={(e) => handleRegenerateThumbnail(e, video)}
                disabled={regenerating === video.id}
                className="p-2 text-gray-400 hover:text-primary-500 transition-colors disabled:opacity-50"
                title={t('videoList.regenerateThumbnail')}
              >
                {regenerating === video.id ? (
                  <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                    />
                  </svg>
                ) : (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                    />
                  </svg>
                )}
              </button>

              {/* Delete button */}
              {onDelete && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(video.id);
                  }}
                  className="p-2 text-gray-400 hover:text-red-500 transition-colors"
                  title={t('videoList.deleteVideo')}
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                    />
                  </svg>
                </button>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};
