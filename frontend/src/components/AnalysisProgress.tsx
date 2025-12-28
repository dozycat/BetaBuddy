import React from 'react';
import { useTranslation } from 'react-i18next';

interface AnalysisProgressProps {
  progress: number;
  currentFrame: number;
  status: string;
}

export const AnalysisProgress: React.FC<AnalysisProgressProps> = ({
  progress,
  currentFrame,
  status,
}) => {
  const { t } = useTranslation();

  return (
    <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-100">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">{t('analysis.inProgress')}</h3>
        <span className="text-sm text-gray-500 capitalize">{status}</span>
      </div>

      <div className="space-y-4">
        {/* Progress bar */}
        <div>
          <div className="flex items-center justify-between text-sm mb-1">
            <span className="text-gray-600">{t('analysis.processingFrames')}</span>
            <span className="font-medium">{progress.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div
              className="bg-primary-500 h-3 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {/* Current frame */}
        <div className="flex items-center gap-3 text-sm text-gray-600">
          <div className="flex items-center gap-2">
            <svg className="animate-spin h-4 w-4 text-primary-500" viewBox="0 0 24 24">
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
                fill="none"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
              />
            </svg>
            <span>{t('analysis.processingFrame', { frame: currentFrame })}</span>
          </div>
        </div>

        {/* Info */}
        <div className="text-xs text-gray-400">
          {t('analysis.detectingPose')}
        </div>
      </div>
    </div>
  );
};
