import type { RefObject } from 'react';
import type { BallObservation } from './session';
import { ballTrail, recentBallTrail } from './ballDetections';
import { useVideoFrameTime } from './useVideoFrameTime';

export function BallTrajectory({ videoRef, observations, fps, threshold, showTrail = true }: { videoRef:RefObject<HTMLVideoElement|null>; observations:BallObservation[]; fps:number; threshold:number; showTrail?:boolean }) {
  const time = useVideoFrameTime(videoRef);
  const current = ballTrail(observations,time,fps,threshold).at(-1);
  const segments = showTrail ? recentBallTrail(observations,time,fps,threshold) : [];
  const opacity = (timestamp: number) => Math.max(0, Math.min(1, 1 - (time - timestamp) / .5));
  return <g className="ball-trajectory" aria-label={current ? 'Ball detected' : 'No confident ball detection'}>
    <g aria-label="Detected positions over the past half second">
      {segments.map(segment => <g key={segment[0].timestamp}>
        {segment.map((observation, i) => {
          const point = observation.position.value;
          const previous = segment[i - 1]?.position.value;
          return <g key={observation.timestamp} opacity={opacity(observation.timestamp)}>
            {previous && <polyline points={`${previous.x},${previous.y} ${point.x},${point.y}`} />}
            <ellipse className="ball-trail-point" cx={point.x} cy={point.y} rx=".002" ry=".0035" />
          </g>;
        })}
      </g>)}
    </g>
    {current && <ellipse cx={current.position.value.x} cy={current.position.value.y} rx=".006" ry=".010" />}
  </g>;
}
