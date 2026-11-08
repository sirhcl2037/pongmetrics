import { useEffect } from 'react';
import type { Point } from './geometry';
import type { Session } from './session';

export type BallLabel = {
  frameIndex: number;
  timestamp: number;
  visibility: 'visible' | 'occluded' | 'out-of-frame' | 'uncertain';
  position: { value: Point; source: 'manual' } | null;
  review?: { action: 'accepted' | 'corrected' | 'visibility'; proposal: Point | null; confidence?: number };
};

export function BallLabeler({ enabled, onToggle, time, fps, video, labels, onLabel, onDelete, onStep,
  proposal, onAccept, onNextUnreviewed, autoAdvance, onAutoAdvance, onGenerate, generating, status, storageStatus }: {
  enabled: boolean; onToggle: () => void; time: number; fps: number;
  video: Session['video'] | null; labels: BallLabel[];
  onLabel: (visibility: BallLabel['visibility']) => void; onDelete: () => void; onStep: (direction: number) => void;
  proposal: { value: Point; confidence?: number } | null; onAccept: () => void; onNextUnreviewed: () => void;
  autoAdvance: boolean; onAutoAdvance: (value: boolean) => void;
  onGenerate: (() => void) | null; generating: boolean; status: string; storageStatus: string;
}) {
  const current = labels.find(label => Math.abs(label.timestamp - time) < .000001);
  useEffect(() => {
    if (!enabled) return;
    const keydown = (event: KeyboardEvent) => {
      const target = event.target;
      if (event.repeat || event.ctrlKey || event.metaKey || event.altKey ||
          (target instanceof HTMLElement && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)))) return;
      const key = event.key.toLowerCase();
      if (!['a', 'o', 'x', 'u', 'n', 'arrowleft', 'arrowright'].includes(key)) return;
      event.preventDefault();
      if (key === 'a') onAccept();
      else if (key === 'o') onLabel('occluded');
      else if (key === 'x') onLabel('out-of-frame');
      else if (key === 'u') onLabel('uncertain');
      else if (key === 'n') onNextUnreviewed();
      else onStep(key === 'arrowleft' ? -1 : 1);
    };
    window.addEventListener('keydown', keydown);
    return () => window.removeEventListener('keydown', keydown);
  }, [enabled, onAccept, onLabel, onNextUnreviewed, onStep]);
  function download() {
    if (!video) return;
    const payload = { format: 'pongmetrics-ball-ground-truth', version: 1, video, fps,
      annotationMode: 'detector-assisted-review',
      coordinateSystem: 'normalized-video-image', frameIndexMeaning: 'round(timestamp * configured FPS); timestamps identify decoded images',
      labels: labels.map(label => ({ ...label, frameIndex: Math.round(label.timestamp * fps) })).sort((a, b) => a.timestamp - b.timestamp) };
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url; link.download = `${video.name}.ball-labels.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return <div className="calibration-controls">
    <div className="workspace-tools">
      <button disabled={!video} aria-pressed={enabled} onClick={onToggle}>Ball review {enabled ? 'on' : 'off'}</button>
      {onGenerate && <button disabled={!video || generating} onClick={onGenerate}>{generating ? 'Detecting… You can return later' : 'Generate review proposals'}</button>}
      <button disabled={!labels.length} onClick={download}>Export ball labels ({labels.length})</button>
    </div>
    {enabled && <>
      <p>Review the yellow proposal. A accepts it; click the actual ball to correct it. O = occluded, X = out of frame, U = uncertain. ← / → steps frames; N skips to the next unreviewed frame. Keep this tab open while detection runs.</p>
      <p>{labels.length} reviewed / approximately {video ? Math.ceil(video.duration * fps) : 0} frames. {storageStatus}</p>
      {status && <p role="status">{status}</p>}
      <div className="workspace-tools">
        <button disabled={!proposal} onClick={onAccept}>Accept proposal (A)</button>
        <label><input type="checkbox" checked={autoAdvance} onChange={event => onAutoAdvance(event.target.checked)} /> Advance after review</label>
        <button onClick={onNextUnreviewed}>Next unreviewed (N)</button>
        <button onClick={() => onStep(-1)}>← Label previous frame</button>
        <button onClick={() => onStep(1)}>Label next frame →</button>
        {(['occluded', 'out-of-frame', 'uncertain'] as const).map(value => <button key={value} aria-pressed={current?.visibility === value} onClick={() => onLabel(value)}>{value}</button>)}
        <button disabled={!current} onClick={onDelete}>Delete current label</button>
      </div>
      <p role="status">Frame ≈ {Math.round(time * fps)} · {time.toFixed(6)} s · {current?.visibility ?? 'unlabeled'}. Unlabeled frames are excluded from evaluation.</p>
      {proposal && <p>Proposal score {proposal.confidence?.toFixed(2) ?? 'unavailable'} · yellow = awaiting review, red = reviewed position.</p>}
      {!proposal && <p>No proposal on this frame. Click the ball if visible, or mark its visibility.</p>}
    </>}
  </div>;
}
