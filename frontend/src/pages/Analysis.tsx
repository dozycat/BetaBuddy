import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { VideoPlayer } from '../components/VideoPlayer';
import { MetricsPanel } from '../components/MetricsPanel';
import { BetaSuggestion } from '../components/BetaSuggestion';
import { AnalysisProgress } from '../components/AnalysisProgress';
import { useWebSocket } from '../hooks/useWebSocket';
import { videoApi, analysisApi } from '../api/client';
import type { Video, AnalysisTask, AnalysisResult } from '../types';

export const Analysis: React.FC = () => {
  const { videoId } = useParams<{ videoId: string }>();
  const navigate = useNavigate();

  const [video, setVideo] = useState<Video | null>(null);
  const [task, setTask] = useState<AnalysisTask | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [currentFrame, setCurrentFrame] = useState(0);

  // WebSocket connection for real-time updates
  const { isConnected } = useWebSocket(
    task?.status === 'processing' ? task.id : null,
    {
      onProgress: (prog, frame) => {
        setProgress(prog);
        setCurrentFrame(frame);
      },
      onComplete: async () => {
        // Fetch the results
        if (videoId) {
          try {
            const analysisResult = await analysisApi.getResults(videoId);
            setResult(analysisResult);
            setTask((prev) => prev ? { ...prev, status: 'completed' } : null);
          } catch (err) {
            console.error('Failed to fetch results:', err);
          }
        }
      },
      onError: (message) => {
        setError(message);
        setTask((prev) => prev ? { ...prev, status: 'failed' } : null);
      },
    }
  );

  // Load video data
  useEffect(() => {
    if (!videoId) return;

    const loadData = async () => {
      setIsLoading(true);
      setError(null);

      try {
        // Load video
        const videoData = await videoApi.get(videoId);
        setVideo(videoData);

        // Try to load existing results
        try {
          const analysisResult = await analysisApi.getResults(videoId);
          setResult(analysisResult);
        } catch {
          // No existing results, that's fine
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load video');
      } finally {
        setIsLoading(false);
      }
    };

    loadData();
  }, [videoId]);

  const handleStartAnalysis = async () => {
    if (!videoId) return;

    setError(null);
    setProgress(0);
    setCurrentFrame(0);

    try {
      const analysisTask = await analysisApi.start(videoId);
      setTask(analysisTask);
      setResult(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start analysis');
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500" />
      </div>
    );
  }

  if (error || !video) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-500 mb-4">{error || 'Video not found'}</p>
          <button
            onClick={() => navigate('/')}
            className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600"
          >
            Back to Home
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => navigate('/')}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M15 19l-7-7 7-7"
                  />
                </svg>
              </button>
              <div>
                <h1 className="text-xl font-bold text-gray-900">{video.original_filename}</h1>
                <p className="text-sm text-gray-500">Video Analysis</p>
              </div>
            </div>

            {!task?.status || task.status === 'completed' || task.status === 'failed' ? (
              <button
                onClick={handleStartAnalysis}
                className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors"
              >
                {result ? 'Re-analyze' : 'Start Analysis'}
              </button>
            ) : null}
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Video player */}
          <div>
            <VideoPlayer src={video.preview_url || `/uploads/${video.filename}`} />

            {/* Video info */}
            <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
              <div className="bg-white rounded-lg p-3 shadow-sm">
                <div className="text-gray-500">Duration</div>
                <div className="font-semibold">
                  {video.duration ? `${video.duration.toFixed(1)}s` : '--'}
                </div>
              </div>
              <div className="bg-white rounded-lg p-3 shadow-sm">
                <div className="text-gray-500">Resolution</div>
                <div className="font-semibold">
                  {video.width && video.height ? `${video.width}x${video.height}` : '--'}
                </div>
              </div>
              <div className="bg-white rounded-lg p-3 shadow-sm">
                <div className="text-gray-500">FPS</div>
                <div className="font-semibold">{video.fps?.toFixed(1) || '--'}</div>
              </div>
            </div>
          </div>

          {/* Analysis section */}
          <div className="space-y-6">
            {/* Show progress if analyzing */}
            {task?.status === 'processing' && (
              <AnalysisProgress
                progress={progress}
                currentFrame={currentFrame}
                status={task.status}
              />
            )}

            {/* Show error if failed */}
            {task?.status === 'failed' && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <p className="text-red-700">{task.error_message || 'Analysis failed'}</p>
              </div>
            )}

            {/* Show results if available */}
            {result && (
              <>
                <MetricsPanel result={result} />
                <BetaSuggestion
                  videoId={video.id}
                  initialSuggestion={result.beta_suggestion}
                />
              </>
            )}

            {/* Show placeholder if no analysis yet */}
            {!task && !result && (
              <div className="bg-white rounded-lg p-8 shadow-sm border border-gray-100 text-center">
                <div className="text-4xl mb-4">🧗</div>
                <h3 className="text-lg font-semibold mb-2">Ready to Analyze</h3>
                <p className="text-gray-600 mb-4">
                  Click "Start Analysis" to begin pose detection and biomechanics calculation
                </p>
                <button
                  onClick={handleStartAnalysis}
                  className="px-6 py-3 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors"
                >
                  Start Analysis
                </button>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
};
