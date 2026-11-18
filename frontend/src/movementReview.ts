import type { TrackingFrame } from './ballDetections.ts';
import type { Point } from './geometry.ts';
import type { BodyLandmark, PlayerBox, PlayerTracks } from './playerTracks.ts';
import type { Session } from './session.ts';

export function inspectPlayerView(tracks: PlayerTracks) {
  const scores = new Map<number, { frames: number; heightSum: number }>();
  for (const people of tracks.frames.values()) {
    for (const person of people) {
      const score = scores.get(person.trackId) ?? { frames: 0, heightSum: 0 };
      score.frames++;
      score.heightSum += person.box.height;
      scores.set(person.trackId, score);
    }
  }
  const [primary] = [...scores].sort((a, b) => b[1].heightSum - a[1].heightSum);
  if (!primary || primary[1].frames < .4 * tracks.frames.size) return { trackId: null, onePlayer: false };

  const [trackId, score] = primary;
  const overlap = new Map<number, number>();
  for (const people of tracks.frames.values()) {
    const player = people.find(person => person.trackId === trackId);
    if (!player) continue;
    for (const person of people) {
      const { x, width, height } = person.box;
      if (person.trackId !== trackId && height >= .4 * player.box.height && x + width / 2 > .05 && x + width / 2 < .95) {
        overlap.set(person.trackId, (overlap.get(person.trackId) ?? 0) + 1);
      }
    }
  }
  return { trackId, onePlayer: ![...overlap.values()].some(frames => frames >= .15 * score.frames) };
}

function joint(person: PlayerBox, name: BodyLandmark['name']) {
  const position = person.landmarks?.find(landmark => landmark.name === name)?.position;
  return position && (position.confidence ?? 0) >= .5 ? position.value : null;
}

function wristRelativeToShoulder(person: PlayerBox, side: 'left' | 'right') {
  const wrist = joint(person, `${side}-wrist`), shoulder = joint(person, `${side}-shoulder`);
  if (!wrist || !shoulder || Math.min(wrist.x, wrist.y, 1 - wrist.x, 1 - wrist.y) < .02) return null;
  return { x: wrist.x - shoulder.x, y: wrist.y - shoulder.y };
}

function ankle(person: PlayerBox, side: 'left' | 'right') {
  const point = joint(person, `${side}-ankle`);
  const { y, height } = person.box;
  return point && y + height < .98 && point.y >= y + .6 * height && point.y < .98 ? point : null;
}

function spread(points: (Point | null)[], height: number, aspect: number) {
  let largest = 0;
  for (const point of points) {
    if (!point) return null;
    for (const other of points) {
      if (!other) return null;
      largest = Math.max(largest, Math.hypot((point.x - other.x) * aspect, point.y - other.y));
    }
  }
  return largest / height;
}

type MovementWindow = { start: number; end: number; lateral: number; swing: number | null; feet: number | null };

export function movementReview(tracks: PlayerTracks, trackId: number,
  video: Pick<Session['video'], 'width' | 'height' | 'duration'>, diagnostics: TrackingFrame[]) {
  const stride = Math.max(1, Math.round(.1 * tracks.fps));
  const length = 4 * stride;
  const aspect = video.width / video.height;
  const startFrame = Math.round(tracks.firstTimestamp * tracks.fps);
  const endFrame = Math.floor(video.duration * tracks.fps);
  const cuts = diagnostics.filter(row => row.sceneCut);
  let tested = 0, cutIndex = 0;
  const windows: MovementWindow[] = [];
  for (let start = startFrame; start + length < endFrame; start += length) {
    tested++;
    const startTime = start / tracks.fps, endTime = (start + length) / tracks.fps;
    while (cuts[cutIndex]?.timestamp <= startTime) cutIndex++;
    if (cuts[cutIndex]?.timestamp <= endTime) continue;
    const samples = [];
    for (let offset = 0; offset <= length; offset += stride) {
      const person = tracks.frames.get(start + offset)?.find(candidate => candidate.trackId === trackId);
      if (!person) break;
      samples.push(person);
    }
    if (samples.length !== 5) continue;
    const heights = samples.map(person => person.box.height).sort((a, b) => a - b);
    const height = heights[2];
    if (heights[4] > 1.45 * heights[0]) continue;
    const centers = samples.map(({ box }) => box.x + box.width / 2);
    function poseRange(point: (person: PlayerBox, side: 'left' | 'right') => Point | null) {
      const left = spread(samples.map(person => point(person, 'left')), height, aspect);
      const right = spread(samples.map(person => point(person, 'right')), height, aspect);
      return left === null ? right : right === null ? left : Math.max(left, right);
    }
    const swing = poseRange(wristRelativeToShoulder), feet = poseRange(ankle);
    windows.push({ start: startTime, end: endTime, swing, feet,
      lateral: (Math.max(...centers) - Math.min(...centers)) * aspect / height });
  }
  return { tested, windows };
}
