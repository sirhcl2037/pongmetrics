import { useMemo, useState, type RefObject } from 'react';
import { automaticRecovery, type RecoveryExample } from './automaticRecovery';
import type { TrackingFrame } from './ballDetections';
import type { PlayerTracks } from './playerTracks';
import type { BallObservation, Session } from './session';
import { formatVideoTime } from './videoFrames';
import { ShotStepper } from './ShotStepper';
import { movementReview, type inspectPlayerView } from './movementReview';
import { MovementPreview } from './MovementPreview';
import type { useRangeReplay } from './useRangeReplay';
import type { RallyRange } from './rallies';
import type { ShotCandidate } from './shotDetection';

export type CoachMode = 'auto' | 'movement' | 'match';

const contactEvidence: Record<ShotCandidate['evidence'], string> = {
  'wrist-proximity': 'The ball changed direction near the tracked wrist.',
  'player-proximity': 'The ball changed direction within reach of one tracked player.',
  'ball-reversal': 'The ball changed direction. The hitter is unclear.',
  'short-ball-gap': 'The ball was missing around contact. Timing comes from the flight on either side of the gap.',
};

function describeMotion(example: RecoveryExample) {
  if (!example.motion) return `Track ${example.trackId} had insufficient stable tracking for a movement estimate.`;
  const { horizontalWidths, seconds } = example.motion;
  const percent = Math.round(Math.abs(horizontalWidths) * 100);
  const direction = horizontalWidths < 0 ? 'left' : 'right';
  const shift = percent < 1 ? 'shifted horizontally by less than 1% of the box width'
    : `shifted ${direction} on screen by about ${percent}% of the box width`;
  return `Track ${example.trackId}’s box center ${shift} over the final ${Math.round(seconds * 1000)} ms.`;
}

type Props = {
  video: Session['video'] | null;
  videoRef: RefObject<HTMLVideoElement | null>;
  time: number;
  onSeek: (time: number) => void;
  observations: BallObservation[];
  fps: number;
  tracks: PlayerTracks | null;
  view: ReturnType<typeof inspectPlayerView> | null;
  diagnostics: TrackingFrame[];
  onAnalyze: (() => void) | null;
  onCancel: (() => void) | null;
  disabled: boolean;
  status: string;
  ballUnavailable: boolean;
  mode: CoachMode;
  onMode: (mode: CoachMode) => void;
  replay: ReturnType<typeof useRangeReplay>;
};

export function AutomaticCoach({ video, videoRef, time, onSeek, observations, fps, tracks, view, diagnostics, onAnalyze, onCancel, disabled, status, ballUnavailable, mode, onMode, replay }: Props) {
  const [playbackError, setPlaybackError] = useState(false);
  const movementFirst = mode === 'movement' || (mode === 'auto' && (view?.onePlayer || ballUnavailable));
  const report = useMemo(() => video && observations.length && !movementFirst
    ? automaticRecovery(observations, fps, tracks, diagnostics) : null, [video, observations, fps, tracks, diagnostics, movementFirst]);
  const exchanges = useMemo(() => {
    const examples = report?.players.flatMap(player => player.examples) ?? [];
    return {
      incoming: new Map(examples.map(example => [example.nextContact, example])),
      outgoing: new Map(examples.map(example => [example.contact, example])),
      measured: examples.filter(example => example.motion !== null).length,
    };
  }, [report]);
  const supportedMatch = report?.players.some(player => player.examples.length) ?? false;
  const movementOnly = !!tracks && (movementFirst || (mode === 'auto' && !!report && !supportedMatch));
  const unsupportedMatch = mode === 'match' && !!tracks && !supportedMatch;
  const showShots = !!report && !movementOnly && !unsupportedMatch;
  const movement = useMemo(() => movementOnly && video && tracks && view?.trackId != null
    ? movementReview(tracks, view.trackId, video, diagnostics) : null, [movementOnly, video, tracks, view, diagnostics]);

  function jump(time: number) {
    const video = videoRef.current;
    if (!video) return;
    setPlaybackError(false);
    replay.jump(time);
    onSeek(video.currentTime);
  }

  function playEvidence(range: RallyRange) {
    videoRef.current?.scrollIntoView({ block: 'nearest' });
    return replay.play(range, { slow: true });
  }

  async function playExchange(example: RecoveryExample) {
    setPlaybackError(false);
    if (!await playEvidence({ start: example.contact, end: example.nextContact })) setPlaybackError(true);
  }

  function renderContact(step: ShotCandidate | undefined) {
    if (!step) return <p>Choose a shot step to see its evidence and replay.</p>;
    const before = exchanges.incoming.get(step.timestamp);
    const after = exchanges.outgoing.get(step.timestamp);
    return <div className="recovery-card" aria-label="Selected contact evidence">
      <strong>Estimated contact · {formatVideoTime(step.timestamp)} · {step.trackId === null ? 'Hitter unclear' : `Track ${step.trackId}`}</strong>
      <p>{contactEvidence[step.evidence]}</p>
      {before && <p><strong>Leading into this contact.</strong> {describeMotion(before)}</p>}
      {after && <p><strong>Leading into the next opponent contact.</strong> {describeMotion(after)}</p>}
      {!before && !after && <p>There’s too little evidence linking this step to a neighboring opponent contact to measure recovery.</p>}
      {(before || after) && <div className="workspace-tools">
        {before && <button onClick={() => void playExchange(before)}>Replay into this contact</button>}
        {after && <button onClick={() => void playExchange(after)}>Replay to next contact</button>}
      </div>}
      {playbackError && <p role="status">Playback could not start. Try replaying the exchange.</p>}
    </div>;
  }

  return <section className="rally-review" aria-labelledby="automatic-coach-heading">
    <h3 id="automatic-coach-heading">Coach preview</h3>
    <p>Analyze a short, fixed-camera clip for shot steps or movement review.</p>
    <div className="workspace-tools">
      <label className="recovery-player">Analysis mode <select disabled={disabled} value={mode}
        onChange={event => onMode(event.target.value as CoachMode)}>
        <option value="auto">Automatic</option>
        <option value="movement">Single-player movement</option>
        <option value="match">Match analysis</option>
      </select></label>
      {onAnalyze && <button disabled={disabled || !video} onClick={onAnalyze}>Analyze clip</button>}
      {onCancel && <button onClick={onCancel}>Cancel analysis</button>}
    </div>
    {status && <p role="status">{status}</p>}
    {!onAnalyze && !report && !movementOnly && <p>Import matching tracks in Advanced tools. Player tracks can show movement; ball detections help estimate contacts.</p>}
    {movementOnly && <MovementPreview trackId={view?.trackId ?? null} report={movement}
      selected={mode === 'movement'} onePlayer={view?.onePlayer ?? false} onReplay={playEvidence} />}
    {unsupportedMatch && <p>Match analysis needs a supported sequence between two players. Use Single-player movement for a one-sided clip.</p>}
    {showShots && <ShotStepper steps={report.turns} time={time} readTime={() => videoRef.current?.currentTime ?? time} onJump={jump} renderContact={renderContact} />}
    {showShots && !tracks && <p>Import matching player tracks in Advanced tools to show estimated hitters and movement.</p>}
    {showShots && tracks && <>
      <p>Movement before the next opponent contact was measurable after {exchanges.measured} of {report.turns.length} estimated contacts. Replays run at half speed.</p>
      {!!report.unassignedContacts && <p>{report.unassignedContacts} contact estimates without a clear hitter.</p>}
    </>}
  </section>;
}
