import type { BallLabel } from './BallLabeler';
import { defaultPlayers, parseSession, type Session } from './session.ts';

export type ReviewDraft = {
  version: 1; video: Session['video']; fps: number; ballFps: number;
  labels: BallLabel[]; observations: Session['ballObservations']; time: number;
};

export function reviewDraftKey(file: Pick<File, 'name' | 'size' | 'lastModified'>) {
  return `pongmetrics-ball-review-v1:${file.name}:${file.size}:${file.lastModified}`;
}

export function parseReviewDraft(text: string, file: Pick<File, 'name' | 'size'>): ReviewDraft {
  const draft = JSON.parse(text) as ReviewDraft;
  if (!draft || draft.version !== 1 || draft.video?.name !== file.name || draft.video?.size !== file.size ||
      !Array.isArray(draft.labels) || !Number.isFinite(draft.time) || draft.time < 0) throw new Error('Invalid review draft.');
  parseSession(JSON.stringify({ version:2, video:draft.video, fps:draft.fps, ballFps:draft.ballFps,
    coordinateSystem:'normalized-video-image', calibration:null, shots:[], players:defaultPlayers(),
    selectedPlayerId:'player-1', ballObservations:draft.observations }));
  const timestamps = new Set<number>();
  for (const label of draft.labels) {
    if (!label || !Number.isFinite(label.timestamp) || label.timestamp < 0 || label.timestamp > draft.video.duration ||
        timestamps.has(label.timestamp) || !Number.isSafeInteger(label.frameIndex) || label.frameIndex < 0 ||
        !['visible', 'occluded', 'out-of-frame', 'uncertain'].includes(label.visibility)) throw new Error('Invalid saved label.');
    timestamps.add(label.timestamp);
    const point = label.position?.value;
    if (label.visibility === 'visible' ? !point || label.position?.source !== 'manual' ||
        !Number.isFinite(point.x) || !Number.isFinite(point.y) || point.x < 0 || point.x > 1 || point.y < 0 || point.y > 1 :
        label.position !== null) throw new Error('Invalid saved ball position.');
  }
  draft.observations.sort((a, b) => a.timestamp-b.timestamp);
  return draft;
}
