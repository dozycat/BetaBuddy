import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { VideoUploader } from '../components/VideoUploader';
import { VideoList } from '../components/VideoList';
import { videoApi } from '../api/client';
import type { Video } from '../types';

export const Home: React.FC = () => {
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
    if (!confirm('Are you sure you want to delete this video?')) {
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
          <h1 className="text-3xl font-bold text-gray-900">BetaBuddy</h1>
          <p className="mt-1 text-gray-500">AI-powered climbing video analysis</p>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Upload section */}
          <div>
            <h2 className="text-xl font-semibold mb-4">Upload Video</h2>
            <VideoUploader onUploadComplete={handleUploadComplete} />
          </div>

          {/* Video list section */}
          <div>
            <h2 className="text-xl font-semibold mb-4">Your Videos</h2>
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
          <h2 className="text-xl font-semibold mb-6">Features</h2>
          <div className="grid md:grid-cols-3 gap-6">
            <FeatureCard
              icon="🦴"
              title="Pose Detection"
              description="33-point body keypoint detection using MediaPipe for accurate movement tracking"
            />
            <FeatureCard
              icon="⚖️"
              title="Biomechanics Analysis"
              description="Calculate center of mass, joint angles, velocity, and acceleration in real-time"
            />
            <FeatureCard
              icon="💡"
              title="AI Beta Suggestions"
              description="Get personalized climbing tips based on your movement patterns"
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
