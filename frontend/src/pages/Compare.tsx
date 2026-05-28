import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { comparisonApi, videoApi } from '../api/client';
import type { MetricComparison, Video, VideoComparisonResponse } from '../types';

const formatValue = (value: number | null, unit = '') => {
  if (value === null || Number.isNaN(value)) {
    return '—';
  }
  return `${value.toFixed(Math.abs(value) >= 10 ? 1 : 3)}${unit ? ` ${unit}` : ''}`;
};

const deltaClass = (metric: MetricComparison) => {
  if (metric.delta === null || metric.interpretation === 'No meaningful change') {
    return 'text-gray-600';
  }
  return metric.interpretation === 'Improved' ? 'text-green-600' : 'text-amber-600';
};

const VideoSelect: React.FC<{
  label: string;
  value: string;
  videos: Video[];
  onChange: (value: string) => void;
}> = ({ label, value, videos, onChange }) => (
  <label className="block">
    <span className="block text-sm font-medium text-gray-700 mb-2">{label}</span>
    <select
      value={value}
      onChange={(event) => onChange(event.target.value)}
      className="w-full rounded-lg border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
    >
      <option value="">Select a video</option>
      {videos.map((video) => (
        <option key={video.id} value={video.id}>
          {video.original_filename}
        </option>
      ))}
    </select>
  </label>
);

export const Compare: React.FC = () => {
  const [videos, setVideos] = useState<Video[]>([]);
  const [baselineVideoId, setBaselineVideoId] = useState('');
  const [comparisonVideoId, setComparisonVideoId] = useState('');
  const [comparison, setComparison] = useState<VideoComparisonResponse | null>(null);
  const [isLoadingVideos, setIsLoadingVideos] = useState(true);
  const [isComparing, setIsComparing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadVideos = useCallback(async () => {
    try {
      const response = await videoApi.list(0, 100);
      setVideos(response.videos);
      if (response.videos.length >= 2) {
        setBaselineVideoId(response.videos[0].id);
        setComparisonVideoId(response.videos[1].id);
      }
    } catch (err) {
      console.error('Failed to load videos:', err);
      setError('Failed to load videos.');
    } finally {
      setIsLoadingVideos(false);
    }
  }, []);

  useEffect(() => {
    loadVideos();
  }, [loadVideos]);

  const canCompare = useMemo(
    () => baselineVideoId && comparisonVideoId && baselineVideoId !== comparisonVideoId,
    [baselineVideoId, comparisonVideoId]
  );

  const handleCompare = async () => {
    if (!canCompare) {
      setError('Choose two different videos.');
      return;
    }

    setIsComparing(true);
    setError(null);

    try {
      const result = await comparisonApi.compare({
        baseline_video_id: baselineVideoId,
        comparison_video_id: comparisonVideoId,
      });
      setComparison(result);
    } catch (err: any) {
      console.error('Failed to compare videos:', err);
      setComparison(null);
      setError(
        err.response?.data?.detail ||
          'Comparison failed. Make sure both videos have completed analysis results.'
      );
    } finally {
      setIsComparing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Auto Beta Comparison</h1>
              <p className="mt-1 text-gray-500">
                Compare two analyzed attempts and find the actionable beta delta.
              </p>
            </div>
            <Link to="/" className="text-primary-600 hover:text-primary-700 font-medium">
              Back to videos
            </Link>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8 space-y-8">
        <section className="bg-white rounded-lg shadow-sm border border-gray-100 p-6">
          <div className="grid md:grid-cols-2 gap-4">
            <VideoSelect
              label="Baseline attempt"
              value={baselineVideoId}
              videos={videos}
              onChange={setBaselineVideoId}
            />
            <VideoSelect
              label="Comparison attempt"
              value={comparisonVideoId}
              videos={videos}
              onChange={setComparisonVideoId}
            />
          </div>

          <div className="mt-6 flex items-center gap-4">
            <button
              type="button"
              onClick={handleCompare}
              disabled={!canCompare || isComparing || isLoadingVideos}
              className="px-5 py-2 rounded-lg bg-primary-600 text-white font-medium hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isComparing ? 'Comparing…' : 'Compare attempts'}
            </button>
            {videos.length < 2 && !isLoadingVideos && (
              <p className="text-sm text-gray-500">Upload at least two videos to compare attempts.</p>
            )}
          </div>

          {error && (
            <div className="mt-4 rounded-lg bg-red-50 border border-red-200 p-4 text-sm text-red-700">
              {error}
            </div>
          )}
        </section>

        {comparison && (
          <>
            <section className="grid lg:grid-cols-2 gap-6">
              {[comparison.baseline_video, comparison.comparison_video].map((video, index) => (
                <div key={video.id} className="bg-white rounded-lg shadow-sm border border-gray-100 p-4">
                  <div className="flex items-start justify-between gap-3 mb-3">
                    <div>
                      <h2 className="font-semibold text-gray-900">
                        {index === 0 ? 'Baseline' : 'Comparison'}
                      </h2>
                      <p className="text-sm text-gray-500">{video.original_filename}</p>
                    </div>
                    <Link
                      to={`/analysis/${video.id}`}
                      className="text-sm text-primary-600 hover:text-primary-700"
                    >
                      Open analysis
                    </Link>
                  </div>
                  {video.annotated_video_url || video.preview_url ? (
                    <video
                      src={video.annotated_video_url || video.preview_url || undefined}
                      controls
                      className="w-full rounded-lg bg-black"
                    />
                  ) : (
                    <div className="aspect-video rounded-lg bg-gray-100 flex items-center justify-center text-gray-500">
                      Video preview unavailable
                    </div>
                  )}
                </div>
              ))}
            </section>

            <section className="grid lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 bg-white rounded-lg shadow-sm border border-gray-100 p-6">
                <h2 className="text-xl font-semibold mb-4">Metric deltas</h2>
                <div className="overflow-x-auto">
                  <table className="min-w-full text-sm">
                    <thead>
                      <tr className="text-left text-gray-500 border-b">
                        <th className="py-2 pr-4">Metric</th>
                        <th className="py-2 pr-4">Baseline</th>
                        <th className="py-2 pr-4">Comparison</th>
                        <th className="py-2 pr-4">Delta</th>
                        <th className="py-2">Read</th>
                      </tr>
                    </thead>
                    <tbody>
                      {comparison.metric_deltas.map((metric) => (
                        <tr key={metric.metric} className="border-b last:border-b-0">
                          <td className="py-3 pr-4 font-medium text-gray-900">{metric.metric}</td>
                          <td className="py-3 pr-4">{formatValue(metric.baseline, metric.unit)}</td>
                          <td className="py-3 pr-4">{formatValue(metric.comparison, metric.unit)}</td>
                          <td className={`py-3 pr-4 font-medium ${deltaClass(metric)}`}>
                            {formatValue(metric.delta, metric.unit)}
                          </td>
                          <td className="py-3 text-gray-600">{metric.interpretation}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow-sm border border-gray-100 p-6">
                <h2 className="text-xl font-semibold mb-4">Coaching notes</h2>
                <ul className="space-y-3">
                  {comparison.coaching_notes.map((note) => (
                    <li key={note} className="flex gap-3 text-sm text-gray-700">
                      <span className="mt-1 h-2 w-2 rounded-full bg-primary-500 flex-shrink-0" />
                      <span>{note}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </section>

            <section className="grid lg:grid-cols-3 gap-6">
              {comparison.phase_comparisons.map((phase) => (
                <div key={phase.phase} className="bg-white rounded-lg shadow-sm border border-gray-100 p-6">
                  <h3 className="font-semibold text-lg text-gray-900 mb-2">{phase.phase}</h3>
                  <p className="text-sm text-gray-600 mb-4">{phase.insight}</p>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-500">Path delta</span>
                      <span className={phase.path_delta > 0 ? 'text-amber-600' : 'text-green-600'}>
                        {formatValue(phase.path_delta, 'units')}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Vertical gain delta</span>
                      <span>{formatValue(phase.vertical_gain_delta, 'units')}</span>
                    </div>
                  </div>
                </div>
              ))}
            </section>

            <section className="bg-white rounded-lg shadow-sm border border-gray-100 p-6">
              <h2 className="text-xl font-semibold mb-4">Movement summary</h2>
              <div className="grid md:grid-cols-4 gap-4 text-sm">
                <SummaryTile label="Baseline movements" value={comparison.movement_summary.baseline_total} />
                <SummaryTile label="Comparison movements" value={comparison.movement_summary.comparison_total} />
                <SummaryTile label="Baseline challenging" value={comparison.movement_summary.baseline_challenging} />
                <SummaryTile label="Comparison challenging" value={comparison.movement_summary.comparison_challenging} />
              </div>
              <div className="mt-4 text-sm text-gray-500 space-y-1">
                {comparison.alignment_notes.map((note) => (
                  <p key={note}>{note}</p>
                ))}
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  );
};

const SummaryTile: React.FC<{ label: string; value: number }> = ({ label, value }) => (
  <div className="rounded-lg bg-gray-50 p-4">
    <div className="text-gray-500">{label}</div>
    <div className="text-2xl font-semibold text-gray-900">{value}</div>
  </div>
);
