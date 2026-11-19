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
    const foregroundArea = player.box.width * player.box.height;
    for (const person of people) {
      const { x, width, height } = person.box;
      const center = x + width / 2;
      // A narrow background box can be tall enough to pass a height-only check.
      if (person.trackId === trackId || height < .4 * player.box.height ||
          width * height < .3 * foregroundArea || center <= .05 || center >= .95) continue;
      overlap.set(person.trackId, (overlap.get(person.trackId) ?? 0) + 1);
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

type Side = 'left' | 'right';
type MovementWindow = {
  start: number; end: number; lateral: number; swing: number | null; feet: number | null;
  lateralDirection?: Side; swingSide?: Side; feetSide?: Side;
};
type MovementMetric = 'lateral' | 'swing' | 'feet';
type PlaybackRange = Pick<MovementWindow, 'start' | 'end'>;
export type MovementExample = PlaybackRange & { peak: PlaybackRange & { size: number; side?: Side } };

export function movementExamples(windows: MovementWindow[], metric: MovementMetric) {
  const ranked = windows.flatMap((window, index) => {
    const value = window[metric];
    return value === null || value < .06 ? [] : [{ index, value }];
  }).sort((a, b) => b.value - a.value);
  const examples: MovementExample[] = [];
  const epsilon = 1e-6;

  function connects(left: number, right: number) {
    const a = windows[left], b = windows[right];
    return !!a && !!b && a[metric] !== null && b[metric] !== null && Math.abs(a.end - b.start) <= epsilon;
  }

  for (const { index, value } of ranked) {
    const threshold = Math.max(.06, .45 * value);
    const peak = windows[index];
    let left = index, right = index;
    // Include nearby activity around the peak, with a cap for continuous drills.
    while (connects(left - 1, left) && windows[left - 1][metric]! >= threshold &&
           peak.start - windows[left - 1].start <= .8 + epsilon) left--;
    while (connects(right, right + 1) && windows[right + 1][metric]! >= threshold &&
           windows[right + 1].end - peak.end <= .8 + epsilon) right++;

    const first = windows[left], last = windows[right];
    const start = connects(left - 1, left) ? Math.max(windows[left - 1].start, first.start - .2) : first.start;
    const end = connects(right, right + 1) ? Math.min(windows[right + 1].end, last.end + .25) : last.end;
    if (examples.some(example => start < example.end + epsilon && end > example.start - epsilon)) continue;
    const side = metric === 'swing' ? peak.swingSide : metric === 'feet' ? peak.feetSide : peak.lateralDirection;
    examples.push({ start, end, peak: { start: peak.start, end: peak.end, size: value, side } });
    if (examples.length === 2) break;
  }
  return examples;
}

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
    function poseRange(point: (person: PlayerBox, side: Side) => Point | null) {
      const left = spread(samples.map(person => point(person, 'left')), height, aspect);
      const right = spread(samples.map(person => point(person, 'right')), height, aspect);
      if (left !== null && (right === null || left >= right)) return { size: left, side: 'left' as const };
      return { size: right, side: right === null ? undefined : 'right' as const };
    }
    const swing = poseRange(wristRelativeToShoulder), feet = poseRange(ankle);
    const left = Math.min(...centers), right = Math.max(...centers);
    windows.push({ start: startTime, end: endTime, swing: swing.size, swingSide: swing.side, feet: feet.size, feetSide: feet.side,
      lateral: (right - left) * aspect / height,
      lateralDirection: centers.indexOf(left) < centers.indexOf(right) ? 'right' : 'left' });
  }
  return { tested, windows };
}
