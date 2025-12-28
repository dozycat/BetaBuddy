import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { betaApi } from '../api/client';

interface BetaSuggestionProps {
  videoId: string;
  initialSuggestion?: string | null;
}

export const BetaSuggestion: React.FC<BetaSuggestionProps> = ({
  videoId,
  initialSuggestion,
}) => {
  const { t } = useTranslation();
  const [suggestion, setSuggestion] = useState<string | null>(initialSuggestion || null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await betaApi.getSuggestion(videoId);
      setSuggestion(result.suggestion);
    } catch (err) {
      setError(err instanceof Error ? err.message : t('beta.failed'));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-gradient-to-br from-blue-50 to-green-50 rounded-lg p-6 shadow-sm border border-gray-100">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <span className="text-2xl">💡</span>
          {t('beta.title')}
        </h3>
        <button
          onClick={handleGenerate}
          disabled={isLoading}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            isLoading
              ? 'bg-gray-200 text-gray-500 cursor-not-allowed'
              : 'bg-primary-500 text-white hover:bg-primary-600'
          }`}
        >
          {isLoading ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
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
              {t('beta.generating')}
            </span>
          ) : suggestion ? (
            t('beta.regenerate')
          ) : (
            t('beta.generate')
          )}
        </button>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 text-red-700 rounded-lg text-sm">
          {error}
        </div>
      )}

      {suggestion ? (
        <div className="prose prose-sm max-w-none">
          <div className="whitespace-pre-wrap text-gray-700 leading-relaxed">
            {suggestion.split('\n').map((line, index) => {
              if (line.startsWith('**') && line.endsWith('**')) {
                return (
                  <h4 key={index} className="font-semibold text-gray-900 mt-4 mb-2">
                    {line.replace(/\*\*/g, '')}
                  </h4>
                );
              }
              if (line.includes('**')) {
                const parts = line.split('**');
                return (
                  <p key={index} className="mb-2">
                    {parts.map((part, i) =>
                      i % 2 === 1 ? (
                        <strong key={i} className="text-gray-900">{part}</strong>
                      ) : (
                        part
                      )
                    )}
                  </p>
                );
              }
              return line ? <p key={index} className="mb-2">{line}</p> : null;
            })}
          </div>
        </div>
      ) : (
        <div className="text-center py-8 text-gray-500">
          <p>{t('beta.placeholder')}</p>
          <p className="text-sm mt-2">{t('beta.basedOn')}</p>
        </div>
      )}
    </div>
  );
};
