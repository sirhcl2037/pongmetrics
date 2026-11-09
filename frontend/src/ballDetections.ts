import { defaultPlayers, parseSession, type BallObservation, type Session } from './session.ts';

export function parseDetections(text: string, video: Session['video']) {
  const data = JSON.parse(text);
  // Decoder and container durations may differ by a frame or rounding error.
  const durationTolerance = Number.isFinite(data?.fps) && data.fps >= 1 ? Math.max(.1, 1 / data.fps) : .1;
  // MKV headers can underreport duration; retain exact name/size/dimension checks.
  if (data?.format !== 'pongmetrics-ball-detections' || data.version !== 1 || data.coordinateSystem !== 'normalized-video-image' || !data.video ||
      ['name', 'size', 'width', 'height'].some(key => data.video[key] !== video[key as keyof typeof video]) ||
      !Number.isFinite(data.video.duration) || (!/\.mkv$/i.test(video.name) && Math.abs(data.video.duration - video.duration) > durationTolerance)) throw new Error('Detection file must match the loaded video.');
  const validated = parseSession(JSON.stringify({ ...data, version:2, calibration:null, shots:[], players:defaultPlayers(), selectedPlayerId:'player-1' }));
  if (validated.ballObservations.some(o => o.position.source !== 'automatic' || o.position.confidence === undefined)) throw new Error('Detections require automatic source and confidence.');
  const sorted = validated.ballObservations.sort((a,b) => a.timestamp-b.timestamp);
  if (sorted.some((o,i) => i > 0 && o.timestamp === sorted[i-1].timestamp)) throw new Error('Duplicate detection timestamps.');
  return { observations:sorted, fps:validated.fps, duration:validated.video.duration };
}

// A missing frame breaks the trail. Never bridge or interpolate across it.
export function ballTrail(observations: BallObservation[], time: number, fps: number, threshold: number) {
  // Imports are sorted. Binary search keeps playback cheap on long recordings.
  let lo = 0, hi = observations.length;
  while (lo < hi) {
    const mid = (lo+hi) >>> 1;
    if (observations[mid].timestamp <= time-.5/fps) lo = mid+1;
    else hi = mid;
  }
  const frame = lo < observations.length && Math.abs(observations[lo].timestamp-time) < .5/fps && (observations[lo].position.confidence ?? 1) >= threshold ? lo : -1;
  if (frame < 0) return [];
  const trail = [observations[frame]];
  for (let i = frame-1; i >= 0 && trail.length < 6; i--) {
    if (trail[0].timestamp-observations[i].timestamp > 1.1/fps || (observations[i].position.confidence ?? 1) < threshold) break;
    trail.unshift(observations[i]);
  }
  return trail;
}

export type TrackingFrame = { timestamp: number; prediction: { x: number; y: number } | null; uncertaintyRadiusPixels: number | null; reason: string };

// Keep a time-based history, including disconnected observations during gaps.
// Binary search avoids scanning the entire recording on each decoded frame.
export function recentBallTrail(observations: BallObservation[], time: number, fps: number, threshold: number, seconds = .5) {
  let lo = 0, hi = observations.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (observations[mid].timestamp < time - seconds) lo = mid + 1;
    else hi = mid;
  }
  const segments: BallObservation[][] = [];
  let segment: BallObservation[] = [];
  for (let i = lo; i < observations.length && observations[i].timestamp < time + .5 / fps; i++) {
    const observation = observations[i];
    if ((observation.position.confidence ?? 1) < threshold) { segment = []; continue; }
    if (!segment.length || observation.timestamp - segment[segment.length - 1].timestamp > 1.1 / fps) {
      segment = [];
      segments.push(segment);
    }
    segment.push(observation);
  }
  return segments;
}

// Diagnostics are separate from BallObservation and never enter session exports.
export function parseTrackingFrames(text: string): TrackingFrame[] {
  const data = JSON.parse(text);
  if (!Array.isArray(data.trackingFrames)) return [];
  const frames: unknown[] = data.trackingFrames;
  if (frames.some((value, i) => {
    if (!value || typeof value !== 'object') return true;
    const r = value as Partial<TrackingFrame>;
    return !Number.isFinite(r.timestamp) || (r.timestamp as number) < 0 || typeof r.reason !== 'string' ||
      (i > 0 && (typeof (frames[i-1] as Partial<TrackingFrame>)?.timestamp !== 'number' || r.timestamp <= (frames[i-1] as Partial<TrackingFrame>).timestamp!)) ||
      (r.prediction !== null && (!r.prediction || !Number.isFinite(r.prediction.x) || !Number.isFinite(r.prediction.y))) ||
      (r.uncertaintyRadiusPixels !== null && (!Number.isFinite(r.uncertaintyRadiusPixels) || r.uncertaintyRadiusPixels < 0));
  })) {
    throw new Error('Invalid tracking diagnostics.');
  }
  return frames as TrackingFrame[];
}
