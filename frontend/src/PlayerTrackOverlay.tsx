import { playerTrackNames, type PlayerTracks } from './playerTracks';
import type { Player } from './session';

const bones = [[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];

export function PlayerTrackOverlay({ time, tracks, players, showLandmarks = true }: { time: number; tracks: PlayerTracks; players: Player[]; showLandmarks?: boolean }) {
  const people = tracks.frames.get(Math.round(time * tracks.fps)) ?? [];
  const names = playerTrackNames(tracks, players);
  return <g className="player-track-overlay" aria-label="Player tracking overlay">
    {people.map(({ trackId, playerId, box, landmarks }) => {
      const label = names.get(trackId) ?? 'Unassigned';
      const color = playerId === 'near-player' ? '#75f587' : playerId === 'far-player' ? '#60ddff' : '#ffe875';
      return <g key={trackId}>
        <rect x={box.x} y={box.y} width={box.width} height={box.height} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" />
        <text x={box.x} y={Math.max(.035, box.y - .01)} fontSize=".032">{label}</text>
        {showLandmarks && landmarks && <g aria-label={`${label} body landmarks`}>
          {bones.map(([a,b]) => {
            const p = landmarks[a].position, q = landmarks[b].position;
            return p && q && (p.confidence ?? 0) >= .35 && (q.confidence ?? 0) >= .35
              ? <line key={`${a}-${b}`} x1={p.value.x} y1={p.value.y} x2={q.value.x} y2={q.value.y} stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" /> : null;
          })}
          {landmarks.map(joint => joint.position && (joint.position.confidence ?? 0) >= .35 &&
            <ellipse key={joint.name} cx={joint.position.value.x} cy={joint.position.value.y} rx=".003" ry=".005" style={{fill:color}}><title>{joint.name}</title></ellipse>)}
        </g>}
      </g>;
    })}
  </g>;
}
