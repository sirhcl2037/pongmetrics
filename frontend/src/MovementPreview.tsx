import { useMemo, useState } from 'react';
import { movementExamples, type MovementExample, type movementReview } from './movementReview';
import type { RallyRange } from './rallies';
import { formatVideoTime } from './videoFrames';

type Props = {
  trackId: number | null;
  report: ReturnType<typeof movementReview> | null;
  onePlayer: boolean;
  selected: boolean;
  onReplay: (range: RallyRange) => Promise<boolean>;
};

const metrics = [
  { key: 'swing', label: 'Wrist motion', description: 'Wrist movement relative to the shoulder, including tosses and resets.' },
  { key: 'feet', label: 'Foot movement', description: 'Ankle movement on screen.' },
  { key: 'lateral', label: 'Side-to-side movement', description: 'Horizontal movement of the player box.' },
] as const;

function describeExample(metric: typeof metrics[number]['key'], { peak }: MovementExample) {
  const percent = Math.round(peak.size * 100);
  const scale = `about ${percent}% of the on-screen box height`;
  if (metric === 'lateral') return `The player box shifted ${peak.side ? `${peak.side} ` : ''}on screen by ${scale}.`;
  const joint = peak.side ? `Estimated ${peak.side} ${metric === 'swing' ? 'wrist' : 'ankle'}` : metric === 'swing' ? 'Estimated wrist' : 'Estimated ankle';
  return metric === 'swing' ? `${joint} shifted by ${scale} relative to the shoulder.` : `${joint} shifted on screen by ${scale}.`;
}

export function MovementPreview({ trackId, report, onePlayer, selected, onReplay }: Props) {
  const [playbackError, setPlaybackError] = useState(false);
  const summaries = useMemo(() => report && metrics.map(metric => ({
    ...metric,
    usable: report.windows.filter(window => window[metric.key] !== null).length,
    examples: movementExamples(report.windows, metric.key),
  })), [report]);

  async function playExample(range: RallyRange) {
    setPlaybackError(false);
    if (!await onReplay(range)) setPlaybackError(true);
  }

  return <section className="recovery-card" aria-labelledby="movement-preview-heading">
    <h4 id="movement-preview-heading">Movement review</h4>
    <p>{selected ? 'The review follows the main foreground player.' : onePlayer
      ? 'This view supports tracking one foreground player.' : 'This clip has too little evidence for reliable shot timing.'}
      {' '}Ball detections can include balls from nearby tables or background objects.</p>
    {report && summaries && trackId !== null ? <>
      <p>Track {trackId}. Short windows from a fixed camera work best. Cropped or hidden joints can reduce coverage, and pose estimates can still be wrong.</p>
      {summaries.some(({ examples }) => examples.length > 0) && <p>Examples play at half speed with nearby footage for context when tracking supports it.</p>}
      {summaries.map(({ key, label, description, usable, examples }) => (
        <div key={key}>
          <p><strong>{label}</strong> · {usable}/{report.tested} windows measured. {description}</p>
          {!!examples.length && <div>
            {examples.map(example => {
              const range = `${formatVideoTime(example.start)}–${formatVideoTime(example.end)}`;
              const peak = `${formatVideoTime(example.peak.start)}–${formatVideoTime(example.peak.end)}`;
              return <div key={example.start} className="workspace-tools">
                <p><strong>{peak}</strong> · {describeExample(key, example)}</p>
                <button onClick={() => void playExample(example)}
                  aria-label={`Play ${label.toLowerCase()} example ${range} at half speed`}>Play {range}</button>
              </div>;
            })}
          </div>}
        </div>
      ))}
      {playbackError && <p role="status">Playback could not start. Try playing the example again.</p>}
    </> : <p>There’s no continuous player track to review.</p>}
  </section>;
}
