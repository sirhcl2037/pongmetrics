import type { TrackingFrame } from './ballDetections.ts';
import type { PlayerTracks } from './playerTracks.ts';
import type { BallObservation, Session } from './session.ts';
import { detectShotCandidates } from './shotDetection.ts';

export const recoveryMotionSeconds = .12;
const minimumBallScore = .15;
const maximumBallGap = .2;

type RecoveryExample = { contact: number; nextContact: number; boxWidths: number | null };

// A fixed-view singles clip should have one pair in most frames that show
// two or more people. Refuse to pick a side when identities keep changing.
function dominantPair(tracks: PlayerTracks): number[] {
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
  return [...pairs.values()].find(pair => pair.count > multiPersonFrames / 2)?.ids ?? [];
}

function motionBefore(time: number, contact: number, trackId: number, tracks: PlayerTracks, aspect: number) {
  const end = Math.round(time * tracks.fps);
  const start = end - Math.round(recoveryMotionSeconds * tracks.fps);
  if (end - start < 2 || start <= Math.round(contact * tracks.fps)) return null;
  const boxes = [];
  for (let frame = start; frame <= end; frame++) {
    const person = tracks.frames.get(frame)?.find(player => player.trackId === trackId);
    if (!person) return null;
    boxes.push(person.box);
  }
  const first = boxes[0], last = boxes[boxes.length - 1];
  const dx = (last.x + last.width / 2) - (first.x + first.width / 2);
  const dy = (last.y + last.height / 2) - (first.y + first.height / 2);
  return Math.hypot(dx, dy / aspect) / first.width;
}

export function automaticRecovery(observations: BallObservation[], fps: number, tracks: PlayerTracks,
  video: Pick<Session['video'], 'width' | 'height'>, diagnostics: TrackingFrame[]) {
  const trackIds = dominantPair(tracks);
  const byPlayer = trackIds.map(trackId => ({ trackId, contacts: 0, examples: [] as RecoveryExample[] }));
  if (!byPlayer.length) return { players: byPlayer, unassignedContacts: 0 };

  const ball = observations.filter(row => row.position.source === 'automatic' && (row.position.confidence ?? 0) >= minimumBallScore);
  const turns = detectShotCandidates(ball, fps, minimumBallScore, tracks, new Set(trackIds));
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
    player.examples.push({ contact: current.timestamp, nextContact: next.timestamp,
      boxWidths: motionBefore(next.timestamp, current.timestamp, player.trackId, tracks, video.width / video.height) });
  }
  return { players: byPlayer, unassignedContacts };
}
