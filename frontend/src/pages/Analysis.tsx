import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { VideoPlayer } from '../components/VideoPlayer';
import { MetricsPanel } from '../components/MetricsPanel';
import { BetaSuggestion } from '../components/BetaSuggestion';
import { AnalysisProgress } from '../components/AnalysisProgress';
import { MovementTimeline } from '../components/MovementTimeline';
import { LanguageSwitcher } from '../components/LanguageSwitcher';
import { useWebSocket } from '../hooks/useWebSocket';
import { videoApi, analysisApi, movementApi } from '../api/client';
import type { Video, AnalysisTask, AnalysisResult, MovementDetectionResponse } from '../types';

export const Analysis: React.FC = () => {
  const { t } = useTranslation();
  const { videoId } = useParams<{ videoId: string }>();
  const navigate = useNavigate();

  const [video, setVideo] = useState<Video | null>(null);
  const [task, setTask] = useState<AnalysisTask | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [annotatedVideoUrl, setAnnotatedVideoUrl] = useState<string | null>(null);
  const [isGeneratingAnnotation, setIsGeneratingAnnotation] = useState(false);
  const [showAnnotated, setShowAnnotated] = useState(true);

  // Climber metrics for unit conversion
  const [heightM, setHeightM] = useState<string>('');
  const [armSpanM, setArmSpanM] = useState<string>('');
  const [showClimberMetrics, setShowClimberMetrics] = useState(false);

  // Movement detection
  const [movementData, setMovementData] = useState<MovementDetectionResponse | null>(null);
  const [isDetectingMovements, setIsDetectingMovements] = useState(false);

  // WebSocket connection for real-time updates
  // Connect for both 'pending' and 'processing' to catch status transitions
  useWebSocket(
    task?.status === 'pending' || task?.status === 'processing' ? task.id : null,
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

            // Auto-generate annotated video after analysis completes
            if (!analysisResult.annotated_video_url) {
              generateAnnotatedVideo();
            } else {
              setAnnotatedVideoUrl(analysisResult.annotated_video_url);
            }
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
          // Set annotated video URL if available
          if (analysisResult.annotated_video_url) {
            setAnnotatedVideoUrl(analysisResult.annotated_video_url);
          }
        } catch {
          // No existing results, that's fine
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : t('analysis.failed'));
      } finally {
        setIsLoading(false);
      }
    };

    loadData();
  }, [videoId, t]);

  const handleStartAnalysis = async () => {
    if (!videoId) return;

    setError(null);
    setProgress(0);
    setCurrentFrame(0);
    setAnnotatedVideoUrl(null);

    try {
      // Parse climber metrics (convert to numbers, undefined if empty)
      const height = heightM ? parseFloat(heightM) : undefined;
      const armSpan = armSpanM ? parseFloat(armSpanM) : undefined;

      const analysisTask = await analysisApi.start(videoId, height, armSpan);
      setTask(analysisTask);
      setResult(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : t('analysis.failed'));
    }
  };

  const generateAnnotatedVideo = async () => {
    if (!videoId || isGeneratingAnnotation) return;

    setIsGeneratingAnnotation(true);
    try {
      const response = await videoApi.annotate(videoId);
      setAnnotatedVideoUrl(response.annotated_video_url);
    } catch (err) {
      console.error('Failed to generate annotated video:', err);
    } finally {
      setIsGeneratingAnnotation(false);
    }
  };

  const handleDetectMovements = async () => {
    if (!videoId || isDetectingMovements) return;

    setIsDetectingMovements(true);
    try {
      const response = await movementApi.detect(videoId, {
        generate_descriptions: true,
      });
      setMovementData(response);
    } catch (err) {
      console.error('Failed to detect movements:', err);
    } finally {
      setIsDetectingMovements(false);
    }
  };

  // Try to load existing movement data when results are loaded
  useEffect(() => {
    if (!videoId || !result) return;

    const loadMovements = async () => {
      try {
        const response = await movementApi.get(videoId);
        if (response.movements.length > 0) {
          setMovementData(response);
        }
      } catch {
        // No existing movement data, that's fine
      }
    };

    loadMovements();
  }, [videoId, result]);

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
          <p className="text-red-500 mb-4">{error || t('analysis.videoNotFound')}</p>
          <button
            onClick={() => navigate('/')}
            className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600"
          >
            {t('nav.backToHome')}
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
                title={t('nav.backToHome')}
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
                <p className="text-sm text-gray-500">{t('analysis.videoAnalysis')}</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <LanguageSwitcher />
              {!task?.status || task.status === 'completed' || task.status === 'failed' ? (
                <button
                  onClick={handleStartAnalysis}
                  className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors"
                >
                  {result ? t('analysis.reAnalyze') : t('analysis.startAnalysis')}
                </button>
              ) : null}
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Video player */}
          <div>
            {/* Video toggle for annotated/original */}
            {result && (
              <div className="mb-4 flex items-center justify-between bg-white rounded-lg p-3 shadow-sm">
                <div className="flex items-center gap-3">
                  <span className="text-sm font-medium text-gray-700">
                    {t('video.viewMode')}:
                  </span>
                  <div className="flex bg-gray-100 rounded-lg p-1">
                    <button
                      onClick={() => setShowAnnotated(false)}
                      className={`px-3 py-1 text-sm rounded-md transition-colors ${
                        !showAnnotated
                          ? 'bg-white text-primary-600 shadow-sm'
                          : 'text-gray-600 hover:text-gray-900'
                      }`}
                    >
                      {t('video.original')}
                    </button>
                    <button
                      onClick={() => setShowAnnotated(true)}
                      disabled={!annotatedVideoUrl && !isGeneratingAnnotation}
                      className={`px-3 py-1 text-sm rounded-md transition-colors ${
                        showAnnotated && annotatedVideoUrl
                          ? 'bg-white text-primary-600 shadow-sm'
                          : 'text-gray-600 hover:text-gray-900'
                      } ${!annotatedVideoUrl && !isGeneratingAnnotation ? 'opacity-50 cursor-not-allowed' : ''}`}
                    >
                      {t('video.annotated')}
                    </button>
                  </div>
                </div>
                {!annotatedVideoUrl && !isGeneratingAnnotation && (
                  <button
                    onClick={generateAnnotatedVideo}
                    className="px-3 py-1 text-sm bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors"
                  >
                    {t('video.generateAnnotated')}
                  </button>
                )}
                {isGeneratingAnnotation && (
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-500" />
                    {t('video.generatingAnnotated')}
                  </div>
                )}
              </div>
            )}

            <VideoPlayer
              src={
                showAnnotated && annotatedVideoUrl
                  ? annotatedVideoUrl
                  : video.preview_url || `/uploads/${video.filename}`
              }
              key={showAnnotated && annotatedVideoUrl ? 'annotated' : 'original'}
            />

            {/* Video info */}
            <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
              <div className="bg-white rounded-lg p-3 shadow-sm">
                <div className="text-gray-500">{t('metrics.duration')}</div>
                <div className="font-semibold">
                  {video.duration ? `${video.duration.toFixed(1)}s` : '--'}
                </div>
              </div>
              <div className="bg-white rounded-lg p-3 shadow-sm">
                <div className="text-gray-500">{t('metrics.resolution')}</div>
                <div className="font-semibold">
                  {video.width && video.height ? `${video.width}x${video.height}` : '--'}
                </div>
              </div>
              <div className="bg-white rounded-lg p-3 shadow-sm">
                <div className="text-gray-500">{t('metrics.fps')}</div>
                <div className="font-semibold">{video.fps?.toFixed(1) || '--'}</div>
              </div>
            </div>
          </div>

          {/* Analysis section */}
          <div className="space-y-6">
            {/* Show progress if analyzing (pending or processing) */}
            {(task?.status === 'pending' || task?.status === 'processing') && (
              <AnalysisProgress
                progress={progress}
                currentFrame={currentFrame}
                status={task.status}
              />
            )}

            {/* Show error if failed */}
            {task?.status === 'failed' && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <p className="text-red-700">{task.error_message || t('analysis.failed')}</p>
              </div>
            )}

            {/* Show results if available */}
            {result && (
              <>
                <MetricsPanel result={result} />
                <MovementTimeline
                  data={movementData}
                  videoDuration={video.duration || 0}
                  loading={isDetectingMovements}
                  onDetect={handleDetectMovements}
                />
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
                <h3 className="text-lg font-semibold mb-2">{t('analysis.readyToAnalyze')}</h3>
                <p className="text-gray-600 mb-4">
                  {t('analysis.readyDescription')}
                </p>

                {/* Climber Metrics Section */}
                <div className="mb-6">
                  <button
                    onClick={() => setShowClimberMetrics(!showClimberMetrics)}
                    className="text-sm text-primary-600 hover:text-primary-700 flex items-center gap-1 mx-auto"
                  >
                    <svg
                      className={`w-4 h-4 transition-transform ${showClimberMetrics ? 'rotate-180' : ''}`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                    {t('analysis.climberMetrics')}
                  </button>

                  {showClimberMetrics && (
                    <div className="mt-4 p-4 bg-gray-50 rounded-lg text-left">
                      <p className="text-xs text-gray-500 mb-3">
                        {t('analysis.climberMetricsDescription')}
                      </p>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            {t('analysis.height')}
                          </label>
                          <input
                            type="number"
                            step="0.01"
                            min="0.5"
                            max="2.5"
                            placeholder="1.75"
                            value={heightM}
                            onChange={(e) => setHeightM(e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            {t('analysis.armSpan')}
                          </label>
                          <input
                            type="number"
                            step="0.01"
                            min="0.5"
                            max="3.0"
                            placeholder="1.80"
                            value={armSpanM}
                            onChange={(e) => setArmSpanM(e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                <button
                  onClick={handleStartAnalysis}
                  className="px-6 py-3 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors"
                >
                  {t('analysis.startAnalysis')}
                </button>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
};
