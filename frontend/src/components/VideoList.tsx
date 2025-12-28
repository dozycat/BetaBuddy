import React from 'react';
import { useTranslation } from 'react-i18next';
import type { Video } from '../types';

interface VideoListProps {
  videos: Video[];
  onSelect: (video: Video) => void;
  onDelete?: (videoId: string) => void;
  selectedId?: string;
}

export const VideoList: React.FC<VideoListProps> = ({
  videos,
  onSelect,
  onDelete,
  selectedId,
}) => {
  const { t } = useTranslation();

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

  if (videos.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        <p>{t('videoList.noVideos')}</p>
        <p className="text-sm mt-1">{t('videoList.getStarted')}</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {videos.map((video) => (
        <div
          key={video.id}
          className={`p-4 rounded-lg border cursor-pointer transition-all ${
            selectedId === video.id
              ? 'border-primary-500 bg-primary-50'
              : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
          }`}
          onClick={() => onSelect(video)}
        >
          <div className="flex items-center justify-between">
            <div className="flex-1 min-w-0">
              <h4 className="font-medium text-gray-900 truncate">
                {video.original_filename}
              </h4>
              <div className="flex items-center gap-3 mt-1 text-sm text-gray-500">
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
      ))}
    </div>
  );
};
