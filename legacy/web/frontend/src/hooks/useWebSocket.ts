import { useEffect, useRef, useState, useCallback } from 'react';
import type { WSMessage } from '../types';

interface UseWebSocketOptions {
  onProgress?: (progress: number, currentFrame: number) => void;
  onKeypoints?: (frameNumber: number, keypoints: any[], centerOfMass: [number, number]) => void;
  onMetrics?: (frameNumber: number, jointAngles: Record<string, number>, velocity?: [number, number], acceleration?: [number, number]) => void;
  onComplete?: (taskId: string, resultId?: string, summary?: any) => void;
  onError?: (message: string) => void;
}

// Get WebSocket URL based on environment
function getWebSocketUrl(taskId: string): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

  // In development, if we're on a Vite dev server port, use the proxy
  // The Vite proxy will forward /ws/* to the backend
  if (import.meta.env.DEV) {
    // Vite proxy handles forwarding to backend
    return `${protocol}//${window.location.host}/ws/analysis/${taskId}`;
  }

  // In production, connect to the same host
  return `${protocol}//${window.location.host}/ws/analysis/${taskId}`;
}

export function useWebSocket(taskId: string | null, options: UseWebSocketOptions = {}) {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const optionsRef = useRef(options);

  // Keep options ref updated
  useEffect(() => {
    optionsRef.current = options;
  }, [options]);

  const connect = useCallback(() => {
    if (!taskId) return;

    const wsUrl = getWebSocketUrl(taskId);
    console.log('Connecting to WebSocket:', wsUrl);

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const message: WSMessage = JSON.parse(event.data);
        setLastMessage(message);

        const opts = optionsRef.current;

        switch (message.type) {
          case 'progress':
            if (message.data?.progress !== undefined) {
              opts.onProgress?.(message.data.progress, message.data.current_frame || 0);
            }
            break;
          case 'keypoints':
            if (message.data?.keypoints && message.data.frame_number !== undefined) {
              opts.onKeypoints?.(
                message.data.frame_number,
                message.data.keypoints,
                message.data.center_of_mass || [0, 0]
              );
            }
            break;
          case 'metrics':
            if (message.data?.joint_angles !== undefined && message.data.frame_number !== undefined) {
              opts.onMetrics?.(
                message.data.frame_number,
                message.data.joint_angles,
                message.data.velocity,
                message.data.acceleration
              );
            }
            break;
          case 'complete':
            opts.onComplete?.(
              message.data?.task_id || taskId,
              message.data?.result_id,
              message.data?.summary
            );
            break;
          case 'error':
            opts.onError?.(message.message || message.data?.message || 'Unknown error');
            break;
          case 'status':
            // Initial status message, can be used to sync state
            if (message.data?.progress !== undefined) {
              opts.onProgress?.(message.data.progress, 0);
            }
            break;
          case 'pong':
            // Heartbeat response, ignore
            break;
          default:
            console.log('Unknown WebSocket message type:', message.type);
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };

    ws.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);
      setIsConnected(false);

      // Only reconnect if not a normal closure and taskId still exists
      if (event.code !== 1000 && taskId) {
        reconnectTimeoutRef.current = window.setTimeout(() => {
          console.log('Attempting WebSocket reconnection...');
          connect();
        }, 3000);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      // Don't call onError here as onclose will also be triggered
    };

    wsRef.current = ws;
  }, [taskId]);

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounted');
      }
    };
  }, [connect]);

  const sendPing = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'ping' }));
    }
  }, []);

  return {
    isConnected,
    lastMessage,
    sendPing,
  };
}
