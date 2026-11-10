import type { RefObject } from 'react';
import type { TrackingFrame } from './ballDetections';
import { useVideoFrameTime } from './useVideoFrameTime';
import { displayedFrameIndex } from './videoFrames';

export function BallTrackingDebug({ videoRef, frames, fps }: { videoRef: RefObject<HTMLVideoElement | null>; frames: TrackingFrame[]; fps: number }) {
  const time = useVideoFrameTime(videoRef);
  const index = displayedFrameIndex(time,fps);
  let lo = 0, hi = frames.length;
  while (lo < hi) {
    const mid = (lo+hi) >>> 1;
    if (Math.round(frames[mid].timestamp*fps) < index) lo = mid+1;
    else hi = mid;
  }
  const row = frames[lo];
  if (!row || Math.round(row.timestamp*fps) !== index || !row.prediction) return null;
  const { x, y } = row.prediction;
  return <g aria-label="Ball prediction and search uncertainty" stroke="#60ddff" fill="none" strokeWidth="1" vectorEffect="non-scaling-stroke">
    <title>Prediction only: {row.reason}</title>
    <ellipse cx={x} cy={y} rx={(row.uncertaintyRadiusPixels ?? 0)/512} ry={(row.uncertaintyRadiusPixels ?? 0)/288} strokeDasharray="4 4" vectorEffect="non-scaling-stroke" />
    <path d={`M${x-.008},${y}h.016 M${x},${y-.014}v.028`} vectorEffect="non-scaling-stroke" />
  </g>;
}
