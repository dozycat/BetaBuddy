import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { MovementDetectionResponse, DetectedMovement } from '../types';

interface MovementTimelineProps {
  data: MovementDetectionResponse | null;
  videoDuration: number;
  onSeek?: (timestamp: number) => void;
  loading?: boolean;
  onDetect?: () => void;
}

// Movement type colors
const movementColors: Record<string, { bg: string; border: string; text: string }> = {
  side_pull: { bg: 'bg-blue-100', border: 'border-blue-400', text: 'text-blue-700' },
  undercling: { bg: 'bg-purple-100', border: 'border-purple-400', text: 'text-purple-700' },
  gaston: { bg: 'bg-orange-100', border: 'border-orange-400', text: 'text-orange-700' },
  heel_hook: { bg: 'bg-green-100', border: 'border-green-400', text: 'text-green-700' },
  toe_hook: { bg: 'bg-teal-100', border: 'border-teal-400', text: 'text-teal-700' },
  flag: { bg: 'bg-yellow-100', border: 'border-yellow-400', text: 'text-yellow-700' },
  drop_knee: { bg: 'bg-pink-100', border: 'border-pink-400', text: 'text-pink-700' },
  dyno: { bg: 'bg-red-100', border: 'border-red-400', text: 'text-red-700' },
};

const defaultColor = { bg: 'bg-gray-100', border: 'border-gray-400', text: 'text-gray-700' };

export const MovementTimeline: React.FC<MovementTimelineProps> = ({
  data,
  videoDuration,
  onSeek,
  loading = false,
  onDetect,
}) => {
  const { t } = useTranslation();
  const [expandedMovement, setExpandedMovement] = useState<string | null>(null);

  // Format time as MM:SS
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Get color for movement type
  const getMovementColor = (type: string) => {
    return movementColors[type] || defaultColor;
  };

  // Calculate position on timeline
  const getTimelinePosition = (timestamp: number): number => {
    if (videoDuration <= 0) return 0;
    return (timestamp / videoDuration) * 100;
  };

  // Handle movement click
  const handleMovementClick = (movement: DetectedMovement) => {
    if (onSeek) {
      onSeek(movement.start_timestamp);
    }
    setExpandedMovement(
      expandedMovement === `${movement.movement_type}-${movement.start_frame}`
        ? null
        : `${movement.movement_type}-${movement.start_frame}`
    );
  };

  // Empty state with detect button
  if (!data && !loading) {
    return (
      <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-100">
        <div className="text-center">
          <svg
            className="mx-auto h-12 w-12 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
          <h3 className="mt-2 text-sm font-medium text-gray-900">
            {t('movements.noData', '未检测到动作')}
          </h3>
          <p className="mt-1 text-sm text-gray-500">
            {t('movements.detectPrompt', '点击下方按钮检测攀岩技术动作')}
          </p>
          {onDetect && (
            <button
              onClick={onDetect}
              className="mt-4 inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              <svg
                className="-ml-1 mr-2 h-5 w-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                />
              </svg>
              {t('movements.detectButton', '检测技术动作')}
            </button>
          )}
        </div>
      </div>
    );
  }

  // Loading state
  if (loading) {
    return (
      <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-100">
        <div className="flex items-center justify-center">
          <svg
            className="animate-spin h-8 w-8 text-indigo-600"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
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
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
          <span className="ml-3 text-gray-600">
            {t('movements.detecting', '正在检测技术动作...')}
          </span>
        </div>
      </div>
    );
  }

  // No movements detected
  if (!data || data.movements.length === 0) {
    return (
      <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-100">
        <div className="text-center text-gray-500">
          <p>{t('movements.noMovements', '未检测到特定技术动作')}</p>
          {onDetect && (
            <button
              onClick={onDetect}
              className="mt-2 text-indigo-600 hover:text-indigo-800 text-sm"
            >
              {t('movements.retryButton', '重新检测')}
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-100 space-y-4">
      {/* Header with summary */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">
          {t('movements.title', '技术动作检测')}
        </h3>
        <div className="flex items-center gap-4 text-sm">
          <span className="text-gray-600">
            {t('movements.totalCount', '共 {{count}} 个动作', { count: data.total_movements })}
          </span>
          {data.challenging_count > 0 && (
            <span className="text-orange-600 flex items-center gap-1">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
              </svg>
              {t('movements.challengingCount', '{{count}} 个高难度', { count: data.challenging_count })}
            </span>
          )}
        </div>
      </div>

      {/* Summary by type */}
      <div className="flex flex-wrap gap-2">
        {Object.entries(data.summary.by_type).map(([type, count]) => {
          const color = getMovementColor(type.toLowerCase().replace(/[\u4e00-\u9fa5]/g, ''));
          return (
            <span
              key={type}
              className={`px-2 py-1 text-xs rounded-full ${color.bg} ${color.text}`}
            >
              {type}: {count}
            </span>
          );
        })}
      </div>

      {/* Visual Timeline */}
      <div className="relative h-12 bg-gray-100 rounded-lg overflow-hidden">
        {/* Time markers */}
        <div className="absolute inset-0 flex justify-between px-2 text-xs text-gray-400">
          <span>0:00</span>
          <span>{formatTime(videoDuration / 2)}</span>
          <span>{formatTime(videoDuration)}</span>
        </div>

        {/* Movement segments */}
        {data.movements.map((movement, index) => {
          const left = getTimelinePosition(movement.start_timestamp);
          const width = getTimelinePosition(movement.end_timestamp) - left;
          const color = getMovementColor(movement.movement_type);

          return (
            <div
              key={`${movement.movement_type}-${index}`}
              className={`absolute top-2 h-8 rounded cursor-pointer transition-all hover:ring-2 hover:ring-offset-1 ${color.bg} ${color.border} border-l-4`}
              style={{
                left: `${left}%`,
                width: `${Math.max(width, 1)}%`,
                minWidth: '8px',
              }}
              onClick={() => handleMovementClick(movement)}
              title={`${movement.movement_name_cn} (${formatTime(movement.start_timestamp)})`}
            >
              {movement.is_challenging && (
                <span className="absolute -top-1 -right-1 text-yellow-500 text-xs">*</span>
              )}
            </div>
          );
        })}
      </div>

      {/* Movement List */}
      <div className="space-y-2 max-h-64 overflow-y-auto">
        {data.movements.map((movement, index) => {
          const color = getMovementColor(movement.movement_type);
          const isExpanded = expandedMovement === `${movement.movement_type}-${movement.start_frame}`;

          return (
            <div
              key={`${movement.movement_type}-${index}`}
              className={`rounded-lg border ${isExpanded ? 'border-indigo-300' : 'border-gray-200'} overflow-hidden`}
            >
              <div
                className={`flex items-center justify-between p-3 cursor-pointer hover:bg-gray-50 ${color.bg}`}
                onClick={() => handleMovementClick(movement)}
              >
                <div className="flex items-center gap-3">
                  <span className={`font-medium ${color.text}`}>
                    {movement.movement_name_cn}
                  </span>
                  {movement.is_challenging && (
                    <span className="text-yellow-500 text-sm" title="高难度动作">
                      *
                    </span>
                  )}
                  <span className="text-xs text-gray-500">
                    {movement.side_cn}
                  </span>
                </div>
                <div className="flex items-center gap-4 text-sm text-gray-600">
                  <span>{formatTime(movement.start_timestamp)}</span>
                  <span className="text-gray-400">-</span>
                  <span>{formatTime(movement.end_timestamp)}</span>
                  <svg
                    className={`w-4 h-4 transform transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </div>

              {/* Expanded details */}
              {isExpanded && (
                <div className="p-3 bg-white border-t border-gray-100 space-y-2">
                  {/* Description */}
                  {movement.description_cn && (
                    <p className="text-sm text-gray-700">{movement.description_cn}</p>
                  )}

                  {/* Key angles */}
                  {Object.keys(movement.key_angles).length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(movement.key_angles).map(([key, value]) => (
                        <span
                          key={key}
                          className="px-2 py-1 bg-gray-100 rounded text-xs text-gray-600"
                        >
                          {key}: {typeof value === 'number' ? value.toFixed(1) : value}
                          {key.includes('angle') || key.includes('elbow') || key.includes('knee') || key.includes('hip') || key.includes('shoulder') ? '°' : ''}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Confidence */}
                  <div className="flex items-center gap-2 text-xs text-gray-500">
                    <span>{t('movements.confidence', '置信度')}: {(movement.confidence * 100).toFixed(0)}%</span>
                    <div className="flex-1 h-1.5 bg-gray-200 rounded-full">
                      <div
                        className="h-full bg-green-500 rounded-full"
                        style={{ width: `${movement.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Re-detect button */}
      {onDetect && (
        <div className="flex justify-end pt-2 border-t border-gray-100">
          <button
            onClick={onDetect}
            className="text-sm text-indigo-600 hover:text-indigo-800 flex items-center gap-1"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            {t('movements.redetectButton', '重新检测')}
          </button>
        </div>
      )}
    </div>
  );
};
