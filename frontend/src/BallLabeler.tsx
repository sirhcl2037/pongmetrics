import type { Point } from './geometry';
import type { Session } from './session';

export type BallLabel = {
  frameIndex: number;
  timestamp: number;
  visibility: 'visible' | 'occluded' | 'out-of-frame' | 'uncertain';
  position: { value: Point; source: 'manual' } | null;
};

export function BallLabeler({ enabled, onToggle, time, fps, video, labels, onLabel, onDelete, onStep }: {
  enabled: boolean; onToggle: () => void; time: number; fps: number;
  video: Session['video'] | null; labels: BallLabel[];
  onLabel: (visibility: BallLabel['visibility']) => void; onDelete: () => void; onStep: (direction: number) => void;
}) {
  const current = labels.find(label => Math.abs(label.timestamp - time) < .000001);
  function download() {
    if (!video) return;
    const payload = { format: 'pongmetrics-ball-ground-truth', version: 1, video, fps,
      coordinateSystem: 'normalized-video-image', frameIndexMeaning: 'round(timestamp * configured FPS); timestamps identify decoded images',
      labels: [...labels].sort((a, b) => a.timestamp - b.timestamp) };
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url; link.download = `${video.name}.ball-labels.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return <div className="calibration-controls">
    <div className="workspace-tools">
      <button disabled={!video} aria-pressed={enabled} onClick={onToggle}>Experimental ball labeling {enabled ? 'on' : 'off'}</button>
      <button disabled={!labels.length} onClick={download}>Export ball labels ({labels.length})</button>
    </div>
    {enabled && <>
      <p>Pause and click the ball center to mark Visible. Detector overlays are hidden. Set the video FPS below before starting. Export these labels before changing videos or reloading.</p>
      <div className="workspace-tools">
        <button onClick={() => onStep(-1)}>← Label previous frame</button>
        <button onClick={() => onStep(1)}>Label next frame →</button>
        {(['occluded', 'out-of-frame', 'uncertain'] as const).map(value => <button key={value} aria-pressed={current?.visibility === value} onClick={() => onLabel(value)}>{value}</button>)}
        <button disabled={!current} onClick={onDelete}>Delete current label</button>
      </div>
      <p role="status">Frame ≈ {Math.round(time * fps)} · {time.toFixed(6)} s · {current?.visibility ?? 'unlabeled'}. Unlabeled frames are excluded from evaluation.</p>
    </>}
  </div>;
}
