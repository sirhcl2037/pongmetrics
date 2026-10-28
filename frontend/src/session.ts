import { validateCorners, type Point } from './geometry.ts';
export type Observation<T> = { value: T; source: 'manual' | 'automatic'; confidence?: number };
export type BallObservation = { timestamp: number; position: Observation<Point> };
export type Player = { id: string; name: string; association?: { timestamp: number; position: Observation<Point> } };
export const defaultPlayers = (): Player[] => [{ id: 'player-1', name: 'Player 1' }, { id: 'player-2', name: 'Player 2' }];
export type Shot = { id: number; time: number; wing: 'FH' | 'BH'; stroke: 'Loop' | 'Drive' | 'Push' | 'Block' | 'Other'; source: 'manual' | 'automatic'; playerId: string | null; playerIdentity?: Observation<string> };
export type Session = {
  version: 2;
  video: { name: string; size: number; width: number; height: number; duration: number };
  fps: number;
  coordinateSystem: 'normalized-video-image';
  calibration: { corners: Observation<Point>[]; timestamp: number } | null;
  shots: Shot[];
  ballObservations: BallObservation[];
  players: Player[];
  selectedPlayerId: string;
  ballFps?: number;
};

export function parseSession(text: string): Session {
  const s = JSON.parse(text) as Session;
  const finite = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n);
  const observation = (o: Observation<Point>) => o && ['manual', 'automatic'].includes(o.source) && o.value && finite(o.value.x) && finite(o.value.y) && o.value.x >= 0 && o.value.x <= 1 && o.value.y >= 0 && o.value.y <= 1 && (o.confidence === undefined || (finite(o.confidence) && o.confidence >= 0 && o.confidence <= 1));
  if (!s || s.version !== 2 || s.coordinateSystem !== 'normalized-video-image' || !finite(s.fps) || s.fps < 1 || s.fps > 240 || !s.video || typeof s.video.name !== 'string' || ![s.video.size,s.video.width,s.video.height,s.video.duration].every(n => finite(n) && n > 0) || !Array.isArray(s.shots) || !Array.isArray(s.ballObservations)) throw new Error('Invalid v0.2 session file.');
  if (s.calibration !== null) {
    if (!s.calibration || !Array.isArray(s.calibration.corners) || !s.calibration.corners.every(observation) || !finite(s.calibration.timestamp) || s.calibration.timestamp < 0 || s.calibration.timestamp > s.video.duration) throw new Error('Invalid calibration.');
    validateCorners(s.calibration.corners.map(o => o.value));
  }
  const ids = new Set<number>();
  if (s.ballFps !== undefined && (!finite(s.ballFps) || s.ballFps < 1 || s.ballFps > 240)) throw new Error('Invalid detection FPS.');
  // Older v0.2 exports have no player data: preserve their shots as unassigned.
  const legacy = s.players === undefined;
  if (legacy) { s.players = defaultPlayers(); s.selectedPlayerId = s.players[0].id; }
  if (!Array.isArray(s.players) || s.players.length !== 2 || s.players.some(p => !p || typeof p.id !== 'string' || !p.id || typeof p.name !== 'string' || !p.name.trim() || p.name.length > 80) || new Set(s.players.map(p => p.id)).size !== 2 || !s.players.some(p => p.id === s.selectedPlayerId)) throw new Error('Invalid players.');
  for (const p of s.players) if (p.association && (!finite(p.association.timestamp) || p.association.timestamp < 0 || p.association.timestamp > s.video.duration || !observation(p.association.position))) throw new Error('Invalid player association.');
  for (const shot of s.shots) {
    if (legacy && shot) shot.playerId = null;
    if (!shot || !Number.isSafeInteger(shot.id) || shot.id < 0 || ids.has(shot.id) || !finite(shot.time) || shot.time < 0 || shot.time > s.video.duration || !['FH','BH'].includes(shot.wing) || !['Loop','Drive','Push','Block','Other'].includes(shot.stroke) || !['manual','automatic'].includes(shot.source)) throw new Error('Invalid shot data.');
    ids.add(shot.id);
    if (shot.playerId !== null && !s.players.some(p => p.id === shot.playerId)) throw new Error('Unknown shot player.');
    if (shot.playerIdentity && (shot.playerIdentity.value !== shot.playerId || !['manual','automatic'].includes(shot.playerIdentity.source) || (shot.playerIdentity.confidence !== undefined && (!finite(shot.playerIdentity.confidence) || shot.playerIdentity.confidence < 0 || shot.playerIdentity.confidence > 1)))) throw new Error('Invalid player identity observation.');
  }
  if (!s.ballObservations.every(o => o && finite(o.timestamp) && o.timestamp >= 0 && o.timestamp <= s.video.duration && observation(o.position))) throw new Error('Invalid ball observations.');
  return s;
}
