import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { VideoUploader } from '../components/VideoUploader';
import { VideoList } from '../components/VideoList';
import { LanguageSwitcher } from '../components/LanguageSwitcher';
import { videoApi } from '../api/client';
import type { Video } from '../types';

export const Home: React.FC = () => {
  const { t } = useTranslation();
  const [videos, setVideos] = useState<Video[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();

  const loadVideos = useCallback(async () => {
    try {
      const response = await videoApi.list();
      setVideos(response.videos);
    } catch (error) {
      console.error('Failed to load videos:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadVideos();
  }, [loadVideos]);

  const handleUploadComplete = (video: Video) => {
    setVideos((prev) => [video, ...prev]);
    navigate(`/analysis/${video.id}`);
  };

  const handleSelectVideo = (video: Video) => {
    navigate(`/analysis/${video.id}`);
  };

  const handleDeleteVideo = async (videoId: string) => {
    if (!confirm(t('videoList.confirmDelete'))) {
      return;
    }

    try {
      await videoApi.delete(videoId);
      setVideos((prev) => prev.filter((v) => v.id !== videoId));
    } catch (error) {
      console.error('Failed to delete video:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">{t('app.name')}</h1>
              <p className="mt-1 text-gray-500">{t('app.tagline')}</p>
            </div>
            <LanguageSwitcher />
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Upload section */}
          <div>
            <h2 className="text-xl font-semibold mb-4">{t('home.uploadVideo')}</h2>
            <VideoUploader onUploadComplete={handleUploadComplete} />
          </div>

          {/* Video list section */}
          <div>
            <h2 className="text-xl font-semibold mb-4">{t('home.yourVideos')}</h2>
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500" />
              </div>
            ) : (
              <VideoList
                videos={videos}
                onSelect={handleSelectVideo}
                onDelete={handleDeleteVideo}
              />
            )}
          </div>
        </div>

        {/* Features section */}
        <div className="mt-12">
          <h2 className="text-xl font-semibold mb-6">{t('home.features')}</h2>
          <div className="grid md:grid-cols-3 gap-6">
            <FeatureCard
              icon="🦴"
              title={t('features.poseDetection.title')}
              description={t('features.poseDetection.description')}
            />
            <FeatureCard
              icon="⚖️"
              title={t('features.biomechanics.title')}
              description={t('features.biomechanics.description')}
            />
            <FeatureCard
              icon="💡"
              title={t('features.aiSuggestions.title')}
              description={t('features.aiSuggestions.description')}
            />
          </div>
        </div>
      </main>
    </div>
  );
};

const FeatureCard: React.FC<{ icon: string; title: string; description: string }> = ({
  icon,
  title,
  description,
}) => (
  <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-100">
    <div className="text-3xl mb-3">{icon}</div>
    <h3 className="font-semibold text-lg mb-2">{title}</h3>
    <p className="text-gray-600 text-sm">{description}</p>
  </div>
);
