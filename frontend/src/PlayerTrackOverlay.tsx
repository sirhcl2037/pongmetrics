import type { RefObject } from 'react';
import type { PlayerTracks } from './playerTracks';
import { useVideoFrameTime } from './useVideoFrameTime';

export function PlayerTrackOverlay({ videoRef, tracks }: { videoRef: RefObject<HTMLVideoElement | null>; tracks: PlayerTracks }) {
  const time = useVideoFrameTime(videoRef);
  const people = tracks.frames.get(Math.round(time * tracks.fps)) ?? [];
  return <g className="player-track-overlay" aria-label="Player tracking overlay">
    {people.map(({ trackId, playerId, box }) => {
      const label = playerId === 'near-player' ? 'Near player' : playerId === 'far-player' ? 'Far player' : `Track ${trackId}`;
      const color = playerId === 'near-player' ? '#75f587' : playerId === 'far-player' ? '#60ddff' : '#ffe875';
      return <g key={trackId}>
        <rect x={box.x} y={box.y} width={box.width} height={box.height} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" />
        <text x={box.x} y={Math.max(.035, box.y - .01)} fontSize=".032">{label}</text>
      </g>;
    })}
  </g>;
}
