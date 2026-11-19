import type { TrackingFrame } from './ballDetections.ts';
import type { PlayerTracks } from './playerTracks.ts';
import type { BallObservation } from './session.ts';
import { detectShotCandidates } from './shotDetection.ts';

export const recoveryMotionSeconds = .12;
const minimumBallScore = .15;
const maximumBallGap = .2;

export type RecoveryExample = {
  trackId: number;
  contact: number;
  nextContact: number;
  motion: { horizontalWidths: number; seconds: number } | null;
};

// A fixed-view singles clip should have one pair in most frames that show
// two or more people. Refuse to pick a side when identities keep changing.
function dominantPair(tracks: PlayerTracks, ball: BallObservation[], fps: number): number[] {
  const pairs = new Map<string, { ids: number[]; count: number }>();
  let multiPersonFrames = 0;
  for (const people of tracks.frames.values()) {
    if (people.length < 2) continue;
    multiPersonFrames++;
    if (people.length !== 2) continue;
    const ids = people.map(person => person.trackId).sort((a, b) => a - b);
    const key = ids.join(',');
    const entry = pairs.get(key) ?? { ids, count: 0 };
    entry.count++;
    pairs.set(key, entry);
  }
  const clean = [...pairs.values()].find(pair => pair.count > multiPersonFrames / 2);
  if (clean) return clean.ids;

  // When officials or spectators are tracked, use observed turns with a
  // uniquely supported hitter to identify the two active tracks.
  const hits = new Map<number, number>();
  for (const turn of detectShotCandidates(ball, fps, minimumBallScore, tracks)) {
    if (turn.trackId !== null) hits.set(turn.trackId, (hits.get(turn.trackId) ?? 0) + 1);
  }
  const [first, second, third] = [...hits].sort((a, b) => b[1] - a[1]);
  if (!second || (third && second[1] === third[1])) return [];
  const ids = [first[0], second[0]].sort((a, b) => a - b);
  let overlap = 0;
  for (const people of tracks.frames.values()) {
    if (ids.every(id => people.some(person => person.trackId === id))) overlap++;
  }
  return overlap > multiPersonFrames / 2 ? ids : [];
}

function motionBefore(time: number, contact: number, trackId: number, tracks: PlayerTracks) {
  const end = Math.floor(time * tracks.fps + 1e-6);
  const start = Math.ceil((time - recoveryMotionSeconds) * tracks.fps - 1e-6);
  if (end - start < 2 || start <= contact * tracks.fps) return null;
  const boxes = [];
  for (let frame = start; frame <= end; frame++) {
    const person = tracks.frames.get(frame)?.find(player => player.trackId === trackId);
    if (!person || person.box.x < .005 || person.box.x + person.box.width > .995) return null;
    boxes.push(person.box);
  }
  const heights = boxes.map(box => box.height);
  if (Math.max(...heights) > 1.45 * Math.min(...heights)) return null;
  const first = boxes[0], last = boxes[boxes.length - 1];
  const dx = (last.x + last.width / 2) - (first.x + first.width / 2);
  return { horizontalWidths: dx / first.width, seconds: (end - start) / tracks.fps };
}

export function automaticRecovery(observations: BallObservation[], fps: number, tracks: PlayerTracks | null,
  diagnostics: TrackingFrame[]) {
  const ball = observations.filter(row => row.position.source === 'automatic' && (row.position.confidence ?? 0) >= minimumBallScore);
  const trackIds = tracks ? dominantPair(tracks, ball, fps) : [];
  const byPlayer = trackIds.map(trackId => ({ trackId, contacts: 0, examples: [] as RecoveryExample[] }));
  const turns = trackIds.length
    ? detectShotCandidates(ball, fps, minimumBallScore, tracks, new Set(trackIds))
    : detectShotCandidates(ball, fps, minimumBallScore);
  if (!tracks || !byPlayer.length) return { turns, players: byPlayer, unassignedContacts: turns.length };
  const cuts = diagnostics.filter(row => row.sceneCut);
  let ballIndex = 0, cutIndex = 0, unassignedContacts = 0;
  for (let i = 0; i < turns.length; i++) {
    const current = turns[i], next = turns[i + 1];
    const player = byPlayer.find(row => row.trackId === current.trackId);
    if (!player) { unassignedContacts++; continue; }
    player.contacts++;
    if (!next || next.trackId === null || current.trackId === next.trackId ||
        !current.playerAxis?.reverses || !next.playerAxis?.reverses ||
        next.timestamp - current.timestamp > 1.25) continue;

    while (cuts[cutIndex]?.timestamp <= current.timestamp) cutIndex++;
    if (cuts[cutIndex]?.timestamp <= next.timestamp) continue;
    while (ball[ballIndex + 1]?.timestamp <= current.timestamp) ballIndex++;
    let last = ballIndex;
    while (ball[last].timestamp < next.timestamp && ball[last + 1] &&
           ball[last + 1].timestamp - ball[last].timestamp <= maximumBallGap + 1e-9) last++;
    if (ball[last].timestamp < next.timestamp) continue;
    player.examples.push({ trackId: player.trackId, contact: current.timestamp, nextContact: next.timestamp,
      motion: motionBefore(next.timestamp, current.timestamp, player.trackId, tracks) });
  }
  return { turns, players: byPlayer, unassignedContacts };
}
