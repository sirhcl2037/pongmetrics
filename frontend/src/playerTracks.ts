import type { Session } from './session';

type PlayerBox = {
  trackId: number;
  playerId: 'near-player' | 'far-player' | null;
  box: { x: number; y: number; width: number; height: number };
};

export type PlayerTracks = {
  fps: number;
  firstTimestamp: number;
  frames: Map<number, PlayerBox[]>;
};

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
  let firstTimestamp = Infinity;
  for (const row of data.frames) {
    if (!Number.isFinite(row?.timestamp) || row.timestamp < 0 || row.timestamp > video.duration || !Array.isArray(row.people)) {
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
    }
    frames.set(frame, row.people);
    firstTimestamp = Math.min(firstTimestamp, row.timestamp);
  }
  return { fps, firstTimestamp, frames };
}
