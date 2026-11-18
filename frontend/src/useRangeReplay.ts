import { useEffect, useRef, type RefObject } from 'react';
import type { RallyRange, ReviewedRally } from './rallies';

type Replay = RallyRange & { rallyId?: string; previousRate: number | null };

function clearReplay(video: HTMLVideoElement, replay: RefObject<Replay | null>) {
  const current = replay.current;
  replay.current = null;
  if (current?.previousRate != null) video.playbackRate = current.previousRate;
}

function stopAtEnd(video: HTMLVideoElement, replay: RefObject<Replay | null>) {
  const range = replay.current;
  if (!range || video.seeking || video.currentTime < range.end) return;
  clearReplay(video, replay);
  video.pause();
  video.currentTime = range.end;
}

export function useRangeReplay(videoRef: RefObject<HTMLVideoElement | null>, displayedTime: number, rallies: ReviewedRally[]) {
  const replay = useRef<Replay | null>(null);
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const progress = () => stopAtEnd(video, replay);
    const pause = () => { if (video.paused) clearReplay(video, replay); };
    const seek = () => {
      if (replay.current && Math.abs(video.currentTime - replay.current.start) > .001) clearReplay(video, replay);
    };
    video.addEventListener('timeupdate', progress);
    video.addEventListener('pause', pause);
    video.addEventListener('ended', pause);
    video.addEventListener('seeking', seek);
    return () => {
      clearReplay(video, replay);
      video.removeEventListener('timeupdate', progress);
      video.removeEventListener('pause', pause);
      video.removeEventListener('ended', pause);
      video.removeEventListener('seeking', seek);
    };
  }, [videoRef]);

  useEffect(() => {
    const video = videoRef.current;
    if (video) stopAtEnd(video, replay);
  }, [displayedTime, videoRef]);

  useEffect(() => {
    const playing = replay.current;
    const video = videoRef.current;
    if (video && playing?.rallyId !== undefined && !rallies.some(rally =>
      rally.id === playing.rallyId && rally.start <= playing.start && rally.end >= playing.end)) {
      clearReplay(video, replay);
      video.pause();
    }
  }, [rallies, videoRef]);

  function clear() {
    const video = videoRef.current;
    if (video) clearReplay(video, replay);
  }

  function jump(time: number) {
    const video = videoRef.current;
    if (!video) return;
    clearReplay(video, replay);
    video.pause();
    video.currentTime = time;
  }

  async function play(range: RallyRange, options: { slow?: boolean; rallyId?: string } = {}) {
    const video = videoRef.current;
    if (!video) return false;
    clearReplay(video, replay);
    const playing = { ...range, rallyId: options.rallyId, previousRate: options.slow ? video.playbackRate : null };
    if (options.slow) video.playbackRate = .5;
    replay.current = playing;
    video.currentTime = range.start;
    try { await video.play(); }
    catch {
      if (replay.current === playing) {
        clearReplay(video, replay);
        return false;
      }
    }
    return true;
  }

  return { clear, jump, play };
}
