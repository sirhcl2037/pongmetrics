import { useEffect, useState, type RefObject } from 'react';

export function useVideoFrameTime(videoRef: RefObject<HTMLVideoElement | null>) {
  const [time, setTime] = useState(0);
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const update = () => setTime(video.currentTime);
    let id: number | undefined;
    const frame = (_: number, metadata: VideoFrameCallbackMetadata) => {
      setTime(metadata.mediaTime);
      id = video.requestVideoFrameCallback(frame);
    };
    if (typeof video.requestVideoFrameCallback === 'function') {
      id = video.requestVideoFrameCallback(frame);
    } else {
      video.addEventListener('timeupdate', update);
    }
    video.addEventListener('seeked', update);
    update();
    return () => {
      if (id !== undefined) video.cancelVideoFrameCallback(id);
      video.removeEventListener('seeked', update);
      video.removeEventListener('timeupdate', update);
    };
  }, [videoRef]);
  return time;
}
