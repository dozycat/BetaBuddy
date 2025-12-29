import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import type { AnalysisResult } from '../types';

interface MetricsPanelProps {
  result: AnalysisResult;
}

// Quality level types
type QualityLevel = 'excellent' | 'good' | 'average' | 'poor';

// Quality level colors and labels
const qualityConfig: Record<QualityLevel, { color: string; bgColor: string; label: string }> = {
  excellent: { color: 'text-green-600', bgColor: 'bg-green-100', label: 'Excellent' },
  good: { color: 'text-blue-600', bgColor: 'bg-blue-100', label: 'Good' },
  average: { color: 'text-orange-500', bgColor: 'bg-orange-100', label: 'Average' },
  poor: { color: 'text-red-500', bgColor: 'bg-red-100', label: 'Poor' },
};

export const MetricsPanel: React.FC<MetricsPanelProps> = ({ result }) => {
  const { t } = useTranslation();
  const summary = result.summary_stats;

  // Get conversion factor (prefer from result, fallback to summary_stats)
  const metersPerUnit = result.meters_per_unit ?? summary?.meters_per_unit;
  const hasConversion = metersPerUnit != null && metersPerUnit > 0;

  // Helper to convert and format values
  const convertValue = (value: number | undefined, factor: number | undefined): number => {
    if (value === undefined || value === 0) return 0;
    if (factor === undefined) return value;
    return value * factor;
  };

  // Prepare data for trajectory chart
  const trajectoryData = result.com_trajectory?.map((point, index) => ({
    frame: index,
    x: point[0],
    y: point[1],
  })) || [];

  // Enhanced MetricCard with tooltip and quality indicator
  const MetricCard = ({
    label,
    value,
    unit,
    color,
    tooltip,
    quality,
  }: {
    label: string;
    value: number | string;
    unit?: string;
    color?: string;
    tooltip?: string;
    quality?: QualityLevel;
  }) => {
    const [showTooltip, setShowTooltip] = useState(false);
    const qualityStyle = quality ? qualityConfig[quality] : null;

    return (
      <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-100 relative">
        <div className="flex items-center justify-between mb-1">
          <div className="text-sm text-gray-500 flex items-center gap-1">
            {label}
            {tooltip && (
              <button
                className="text-gray-400 hover:text-gray-600 focus:outline-none"
                onMouseEnter={() => setShowTooltip(true)}
                onMouseLeave={() => setShowTooltip(false)}
                onClick={() => setShowTooltip(!showTooltip)}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </button>
            )}
          </div>
          {qualityStyle && (
            <span className={`text-xs px-2 py-0.5 rounded-full ${qualityStyle.bgColor} ${qualityStyle.color}`}>
              {t(`metrics.quality.${quality}`)}
            </span>
          )}
        </div>
        <div className={`text-2xl font-bold ${color || qualityStyle?.color || 'text-gray-900'}`}>
          {typeof value === 'number' ? value.toFixed(1) : value}
          {unit && <span className="text-sm font-normal text-gray-500 ml-1">{unit}</span>}
        </div>
        {/* Tooltip */}
        {tooltip && showTooltip && (
          <div className="absolute z-10 left-0 right-0 top-full mt-2 p-3 bg-gray-800 text-white text-xs rounded-lg shadow-lg">
            <div className="absolute -top-1 left-4 w-2 h-2 bg-gray-800 rotate-45" />
            {tooltip}
          </div>
        )}
      </div>
    );
  };

  // Quality level determination functions
  const getEfficiencyQuality = (score: number): QualityLevel => {
    if (score >= 0.85) return 'excellent';
    if (score >= 0.7) return 'good';
    if (score >= 0.5) return 'average';
    return 'poor';
  };

  const getDynoQuality = (count: number): QualityLevel | undefined => {
    // Dyno moves are technique-dependent, more is not necessarily better
    // Just indicate presence without quality judgment for 0
    if (count === 0) return undefined;
    if (count <= 2) return 'good';
    if (count <= 4) return 'average';
    return 'excellent'; // Many dynos indicates dynamic climbing
  };

  return (
    <div className="space-y-6">
      {/* Summary Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <MetricCard
          label={t('metrics.efficiency')}
          value={(summary?.avg_efficiency || 0) * 100}
          unit="%"
          tooltip={t('metrics.tooltips.efficiency')}
          quality={getEfficiencyQuality(summary?.avg_efficiency || 0)}
        />
        <MetricCard
          label={t('metrics.duration')}
          value={summary?.duration || 0}
          unit="s"
          tooltip={t('metrics.tooltips.duration')}
        />
        <MetricCard
          label={t('metrics.dynoMoves')}
          value={summary?.dyno_count || 0}
          tooltip={t('metrics.tooltips.dyno')}
          quality={getDynoQuality(summary?.dyno_count || 0)}
        />
      </div>

      {/* Additional Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <MetricCard
          label={t('metrics.maxVelocity')}
          value={hasConversion
            ? convertValue(summary?.max_velocity, metersPerUnit)
            : (summary?.max_velocity || 0)
          }
          unit={hasConversion ? 'm/s' : 'units/s'}
          tooltip={t('metrics.tooltips.velocity')}
        />
        <MetricCard
          label={t('metrics.maxAcceleration')}
          value={hasConversion
            ? convertValue(summary?.max_acceleration, metersPerUnit)
            : (summary?.max_acceleration || 0)
          }
          unit={hasConversion ? 'm/s²' : 'units/s²'}
          tooltip={t('metrics.tooltips.acceleration')}
        />
        <MetricCard
          label={t('metrics.totalDistance')}
          value={hasConversion
            ? convertValue(summary?.total_distance, metersPerUnit)
            : (summary?.total_distance || 0)
          }
          unit={hasConversion ? 'm' : 'units'}
          tooltip={t('metrics.tooltips.distance')}
        />
      </div>

      {/* Center of Mass Trajectory Chart */}
      {trajectoryData.length > 0 && (
        <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold mb-4">{t('metrics.comTrajectory')}</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={trajectoryData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="frame" label={{ value: t('metrics.frame'), position: 'bottom' }} />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="x"
                stroke="#3b82f6"
                name={t('metrics.xPosition')}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="y"
                stroke="#f97316"
                name={t('metrics.yPosition')}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};
