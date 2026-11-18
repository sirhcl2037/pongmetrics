import type { movementReview } from './movementReview';
import { formatVideoTime } from './videoFrames';

type Props = {
  trackId: number | null;
  report: ReturnType<typeof movementReview> | null;
  onePlayer: boolean;
  selected: boolean;
  onJump: (time: number) => void;
};

const metrics = [
  { key: 'swing', label: 'Wrist motion', description: 'Wrist movement relative to the shoulder, including tosses and resets.' },
  { key: 'feet', label: 'Foot movement', description: 'Ankle movement on screen.' },
  { key: 'lateral', label: 'Side-to-side movement', description: 'Horizontal movement of the player box.' },
] as const;

export function MovementPreview({ trackId, report, onePlayer, selected, onJump }: Props) {
  return <section className="recovery-card" aria-labelledby="movement-preview-heading">
    <h4 id="movement-preview-heading">Movement review</h4>
    <p>{selected ? 'The review follows the main foreground player.' : onePlayer
      ? 'This view supports tracking one foreground player.' : 'This clip has too little evidence for reliable shot timing.'}
      {' '}Ball detections can include balls from nearby tables or background objects.</p>
    {report && trackId !== null ? <>
      <p>Track {trackId}. Short windows from a fixed camera work best. Cropped or hidden joints can reduce coverage, and pose estimates can still be wrong.</p>
      {metrics.map(({ key, label, description }) => {
        const usable = report.windows.flatMap(window => {
          const value = window[key];
          return value === null ? [] : [{ ...window, value }];
        });
        const ranked = usable.filter(window => window.value >= .06).sort((a, b) => b.value - a.value);
        const strongest = ranked[0];
        // Neighboring windows can both be part of the same movement.
        const second = strongest && ranked.find(window =>
          window.end < strongest.start - 1e-6 || window.start > strongest.end + 1e-6);
        const examples = [strongest, second].filter(example => example !== undefined);
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
