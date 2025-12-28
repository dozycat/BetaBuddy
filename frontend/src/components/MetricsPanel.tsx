import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';
import type { AnalysisResult } from '../types';

interface MetricsPanelProps {
  result: AnalysisResult;
}

export const MetricsPanel: React.FC<MetricsPanelProps> = ({ result }) => {
  const summary = result.summary_stats;
  const jointAngles = result.joint_angle_stats;

  // Prepare data for trajectory chart
  const trajectoryData = result.com_trajectory?.map((point, index) => ({
    frame: index,
    x: point[0],
    y: point[1],
  })) || [];

  // Prepare data for radar chart (joint angles)
  const radarData = jointAngles ? [
    { joint: 'L Elbow', angle: jointAngles.left_elbow?.avg || 0 },
    { joint: 'R Elbow', angle: jointAngles.right_elbow?.avg || 0 },
    { joint: 'L Shoulder', angle: jointAngles.left_shoulder?.avg || 0 },
    { joint: 'R Shoulder', angle: jointAngles.right_shoulder?.avg || 0 },
    { joint: 'L Hip', angle: jointAngles.left_hip?.avg || 0 },
    { joint: 'R Hip', angle: jointAngles.right_hip?.avg || 0 },
    { joint: 'L Knee', angle: jointAngles.left_knee?.avg || 0 },
    { joint: 'R Knee', angle: jointAngles.right_knee?.avg || 0 },
  ] : [];

  const MetricCard = ({ label, value, unit, color }: { label: string; value: number | string; unit?: string; color?: string }) => (
    <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-100">
      <div className="text-sm text-gray-500 mb-1">{label}</div>
      <div className={`text-2xl font-bold ${color || 'text-gray-900'}`}>
        {typeof value === 'number' ? value.toFixed(1) : value}
        {unit && <span className="text-sm font-normal text-gray-500 ml-1">{unit}</span>}
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Summary Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Stability Score"
          value={(summary?.avg_stability_score || 0) * 100}
          unit="%"
          color={(summary?.avg_stability_score || 0) > 0.7 ? 'text-green-600' : 'text-orange-500'}
        />
        <MetricCard
          label="Efficiency"
          value={(summary?.avg_efficiency || 0) * 100}
          unit="%"
          color={(summary?.avg_efficiency || 0) > 0.7 ? 'text-green-600' : 'text-orange-500'}
        />
        <MetricCard
          label="Duration"
          value={summary?.duration || 0}
          unit="s"
        />
        <MetricCard
          label="Dyno Moves"
          value={summary?.dyno_count || 0}
          color={summary?.dyno_count ? 'text-blue-600' : undefined}
        />
      </div>

      {/* Additional Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <MetricCard
          label="Max Velocity"
          value={summary?.max_velocity || 0}
          unit="units/s"
        />
        <MetricCard
          label="Max Acceleration"
          value={summary?.max_acceleration || 0}
          unit="units/s²"
        />
        <MetricCard
          label="Total Distance"
          value={summary?.total_distance || 0}
          unit="units"
        />
      </div>

      {/* Charts */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Center of Mass Trajectory */}
        {trajectoryData.length > 0 && (
          <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-100">
            <h3 className="text-lg font-semibold mb-4">Center of Mass Trajectory</h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={trajectoryData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="frame" label={{ value: 'Frame', position: 'bottom' }} />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="x"
                  stroke="#3b82f6"
                  name="X Position"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="y"
                  stroke="#f97316"
                  name="Y Position"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Joint Angles Radar */}
        {radarData.length > 0 && (
          <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-100">
            <h3 className="text-lg font-semibold mb-4">Average Joint Angles</h3>
            <ResponsiveContainer width="100%" height={250}>
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="joint" tick={{ fontSize: 10 }} />
                <PolarRadiusAxis domain={[0, 180]} />
                <Radar
                  name="Angle"
                  dataKey="angle"
                  stroke="#22c55e"
                  fill="#22c55e"
                  fillOpacity={0.5}
                />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Detailed Joint Angle Stats */}
      {jointAngles && (
        <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold mb-4">Joint Angle Details</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2 px-3">Joint</th>
                  <th className="text-right py-2 px-3">Min</th>
                  <th className="text-right py-2 px-3">Avg</th>
                  <th className="text-right py-2 px-3">Max</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(jointAngles).map(([joint, stats]) => (
                  <tr key={joint} className="border-b border-gray-50">
                    <td className="py-2 px-3 capitalize">{joint.replace('_', ' ')}</td>
                    <td className="text-right py-2 px-3">{stats.min?.toFixed(1)}°</td>
                    <td className="text-right py-2 px-3 font-medium">{stats.avg?.toFixed(1)}°</td>
                    <td className="text-right py-2 px-3">{stats.max?.toFixed(1)}°</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
