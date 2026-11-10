import type { Observation, Player, Session } from './session';
import type { Point } from './geometry';
import { displayedFrameIndex } from './videoFrames.ts';

export const landmarkNames = ['nose','left-eye','right-eye','left-ear','right-ear','left-shoulder','right-shoulder','left-elbow','right-elbow','left-wrist','right-wrist','left-hip','right-hip','left-knee','right-knee','left-ankle','right-ankle'] as const;
export type BodyLandmark = { name: typeof landmarkNames[number]; position: Observation<Point> | null };
export type PlayerBox = {
  trackId: number;
  playerId: 'near-player' | 'far-player' | null;
  box: { x: number; y: number; width: number; height: number };
  landmarks?: BodyLandmark[];
};

export function clickedPlayerTrack(tracks: PlayerTracks, timestamp: number, point: Point): PlayerBox | null {
  const people = tracks.frames.get(displayedFrameIndex(timestamp,tracks.fps)) ?? [];
  const matches = people.filter(({ box }) => point.x >= box.x && point.x <= box.x + box.width &&
    point.y >= box.y && point.y <= box.y + box.height);
  return matches.length === 1 ? matches[0] : null;
}

export type PlayerTracks = {
  json?: string;
  fps: number;
  firstTimestamp: number;
  frames: Map<number, PlayerBox[]>;
};

// Reuse saved image clicks so session imports and later tracking imports also bind names.
export function playerTrackNames(tracks: PlayerTracks, players: Player[]): Map<number, string> {
  const names = new Map<number, string>();
  const claimed = new Set<number>();
  for (const player of players) {
    if (!player.association) continue;
    const { timestamp, position } = player.association;
    const match = clickedPlayerTrack(tracks, timestamp, position.value);
    if (!match) continue;
    const { trackId } = match;
    if (claimed.has(trackId)) {
      names.delete(trackId);
      continue;
    }
    claimed.add(trackId);
    names.set(trackId, player.name);
  }
  return names;
}

export function parsePlayerTracks(text: string, video: Session['video']): PlayerTracks {
  const data = JSON.parse(text);
  if (data?.format !== 'pongmetrics-player-tracks' || data.version !== 1 ||
      data.coordinateSystem !== 'normalized-video-image' || !data.video ||
      data.video.name !== video.name || data.video.size !== video.size ||
      data.video.width !== video.width || data.video.height !== video.height) {
    throw new Error('Load the original video that matches these player tracks.');
  }
  const { fps } = data;
  if (!Number.isFinite(fps) || fps < 1 || fps > 240 || !Array.isArray(data.frames) || !data.frames.length) {
    throw new Error('Invalid player track FPS or frames.');
  }
  const frames = new Map<number, PlayerBox[]>();
  const duration = /\.mkv$/i.test(video.name) && Number.isFinite(data.video.duration) ? Math.max(video.duration,data.video.duration) : video.duration;
  let firstTimestamp = Infinity;
  for (const row of data.frames) {
    if (!Number.isFinite(row?.timestamp) || row.timestamp < 0 || row.timestamp > duration || !Array.isArray(row.people)) {
      throw new Error('Invalid player track frame.');
    }
    const frame = Math.round(row.timestamp * fps);
    if (frames.has(frame)) throw new Error('Duplicate player track frame.');
    const ids = new Set<number>();
    for (const person of row.people) {
      const box = person?.box;
      if (!Number.isSafeInteger(person?.trackId) || ids.has(person.trackId) ||
          ![null, 'near-player', 'far-player'].includes(person.playerId) ||
          !box || ![box.x, box.y, box.width, box.height].every(Number.isFinite) ||
          box.x < 0 || box.y < 0 || box.width <= 0 || box.height <= 0 ||
          box.x + box.width > 1.000001 || box.y + box.height > 1.000001) {
        throw new Error('Invalid player box or identity.');
      }
      ids.add(person.trackId);
      if (person.landmarks !== undefined && (!Array.isArray(person.landmarks) || person.landmarks.length !== 17 ||
          person.landmarks.some((joint: BodyLandmark, index: number) => {
            if (!joint || joint.name !== landmarkNames[index]) return true;
            const p = joint.position;
            return p !== null && (!p || p.source !== 'automatic' || !p.value ||
              ![p.value.x,p.value.y,p.confidence].every(Number.isFinite) || p.value.x < 0 || p.value.x > 1 ||
              p.value.y < 0 || p.value.y > 1 || p.confidence === undefined || p.confidence < 0 || p.confidence > 1);
          }))) throw new Error('Invalid body landmarks.');
    }
    frames.set(frame, row.people);
    firstTimestamp = Math.min(firstTimestamp, row.timestamp);
  }
  return { fps, firstTimestamp, frames, json:text };
}
