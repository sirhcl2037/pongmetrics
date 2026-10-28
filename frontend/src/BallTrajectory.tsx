import { useEffect, useState, type RefObject } from 'react';
import type { BallObservation } from './session';
import { ballTrail } from './ballDetections';

export function BallTrajectory({ videoRef, observations, fps, threshold }: { videoRef:RefObject<HTMLVideoElement|null>; observations:BallObservation[]; fps:number; threshold:number }) {
  const [time,setTime] = useState(0);
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    let id = 0;
    const update = () => setTime(video.currentTime);
    const frame = (_:number, metadata:VideoFrameCallbackMetadata) => {
      setTime(metadata.mediaTime);
      id = video.requestVideoFrameCallback(frame);
    };
    id = video.requestVideoFrameCallback(frame);
    video.addEventListener('seeked',update);
    return () => { video.cancelVideoFrameCallback(id); video.removeEventListener('seeked',update); };
  },[videoRef]);
  const trail = ballTrail(observations,time,fps,threshold);
  const current = trail.at(-1);
  return <g className="ball-trajectory" aria-label={current ? 'Ball detected' : 'No confident ball detection'}>
    {trail.length > 1 && <polyline points={trail.map(o => `${o.position.value.x},${o.position.value.y}`).join(' ')} />}
    {current && <ellipse cx={current.position.value.x} cy={current.position.value.y} rx=".006" ry=".010" />}
  </g>;
}
