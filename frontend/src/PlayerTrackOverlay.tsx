import { playerTrackNames, type PlayerTracks } from './playerTracks';
import type { Player } from './session';

export function PlayerTrackOverlay({ time, tracks, players }: { time: number; tracks: PlayerTracks; players: Player[] }) {
  const people = tracks.frames.get(Math.round(time * tracks.fps)) ?? [];
  const names = playerTrackNames(tracks, players);
  return <g className="player-track-overlay" aria-label="Player tracking overlay">
    {people.map(({ trackId, playerId, box }) => {
      const label = names.get(trackId) ?? 'Unassigned';
      const color = playerId === 'near-player' ? '#75f587' : playerId === 'far-player' ? '#60ddff' : '#ffe875';
      return <g key={trackId}>
        <rect x={box.x} y={box.y} width={box.width} height={box.height} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" />
        <text x={box.x} y={Math.max(.035, box.y - .01)} fontSize=".032">{label}</text>
      </g>;
    })}
  </g>;
}
