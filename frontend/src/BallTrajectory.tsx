import type { RefObject } from 'react';
import type { BallObservation } from './session';
import { ballTrail } from './ballDetections';
import { useVideoFrameTime } from './useVideoFrameTime';

export function BallTrajectory({ videoRef, observations, fps, threshold }: { videoRef:RefObject<HTMLVideoElement|null>; observations:BallObservation[]; fps:number; threshold:number }) {
  const time = useVideoFrameTime(videoRef);
  const trail = ballTrail(observations,time,fps,threshold);
  const current = trail.at(-1);
  return <g className="ball-trajectory" aria-label={current ? 'Ball detected' : 'No confident ball detection'}>
    {trail.length > 1 && <polyline points={trail.map(o => `${o.position.value.x},${o.position.value.y}`).join(' ')} />}
    {current && <ellipse cx={current.position.value.x} cy={current.position.value.y} rx=".006" ry=".010" />}
  </g>;
}
