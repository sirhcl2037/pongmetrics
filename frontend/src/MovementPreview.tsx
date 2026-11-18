import type { movementReview } from './movementReview';
import { formatVideoTime } from './videoFrames';

type Props = {
  trackId: number | null;
  report: ReturnType<typeof movementReview> | null;
  onePlayer: boolean;
  onJump: (time: number) => void;
};

const metrics = [
  { key: 'swing', label: 'Arm swing', description: 'Wrist movement relative to the shoulder.' },
  { key: 'feet', label: 'Foot movement', description: 'Ankle movement on screen.' },
  { key: 'lateral', label: 'Side-to-side movement', description: 'Horizontal movement of the player box.' },
] as const;

export function MovementPreview({ trackId, report, onePlayer, onJump }: Props) {
  return <section className="recovery-card" aria-labelledby="movement-preview-heading">
    <h4 id="movement-preview-heading">Movement review</h4>
    <p>{onePlayer ? 'Only one player could be tracked reliably.' : 'This clip has too little evidence for reliable shot timing.'}
      {' '}Ball detections can be wrong in this view, so the review uses player motion.</p>
    {report && trackId !== null ? <>
      <p>Track {trackId}. Short windows from a fixed camera work best. Cropped or hidden joints can reduce coverage, and pose estimates can still be wrong.</p>
      {metrics.map(({ key, label, description }) => {
        const usable = report.windows.flatMap(window => {
          const value = window[key];
          return value === null ? [] : [{ ...window, value }];
        });
        const examples = usable.filter(window => window.value >= .06)
          .sort((a, b) => b.value - a.value).slice(0, 2);
        return <div key={key}>
          <p><strong>{label}</strong> · {usable.length}/{report.tested} windows measured. {description}</p>
          {!!examples.length && <div className="workspace-tools">
            {examples.map(({ start, end }) => <button key={start} onClick={() => onJump(start)}
              aria-label={`Jump to ${label.toLowerCase()} example at ${formatVideoTime(start)}`}>
              {formatVideoTime(start)}–{formatVideoTime(end)}
            </button>)}
          </div>}
        </div>;
      })}
    </> : <p>There’s no continuous player track to review.</p>}
  </section>;
}
