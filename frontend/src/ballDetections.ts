import { defaultPlayers, parseSession, type BallObservation, type Session } from './session.ts';

export function parseDetections(text: string, video: Session['video']) {
  const data = JSON.parse(text);
  if (data?.format !== 'pongmetrics-ball-detections' || data.version !== 1 || data.coordinateSystem !== 'normalized-video-image' || !data.video || (Object.keys(video) as (keyof typeof video)[]).some(key => data.video[key] !== video[key])) throw new Error('Detection file must match the loaded video.');
  const validated = parseSession(JSON.stringify({ ...data, version:2, calibration:null, shots:[], players:defaultPlayers(), selectedPlayerId:'player-1' }));
  if (validated.ballObservations.some(o => o.position.source !== 'automatic' || o.position.confidence === undefined)) throw new Error('Detections require automatic source and confidence.');
  const sorted = validated.ballObservations.sort((a,b) => a.timestamp-b.timestamp);
  if (sorted.some((o,i) => i > 0 && o.timestamp === sorted[i-1].timestamp)) throw new Error('Duplicate detection timestamps.');
  return { observations:sorted, fps:validated.fps };
}

// A missing frame breaks the trail. Never bridge or interpolate across it.
export function ballTrail(observations: BallObservation[], time: number, fps: number, threshold: number) {
  const frame = observations.findIndex(o => Math.abs(o.timestamp-time) < .5/fps && (o.position.confidence ?? 1) >= threshold);
  if (frame < 0) return [];
  const trail = [observations[frame]];
  for (let i = frame-1; i >= 0 && trail.length < 6; i--) {
    if (trail[0].timestamp-observations[i].timestamp > 1.1/fps || (observations[i].position.confidence ?? 1) < threshold) break;
    trail.unshift(observations[i]);
  }
  return trail;
}
