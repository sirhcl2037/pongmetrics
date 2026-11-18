import { useMemo, type RefObject } from 'react';
import { automaticRecovery, recoveryMotionSeconds } from './automaticRecovery';
import type { TrackingFrame } from './ballDetections';
import type { PlayerTracks } from './playerTracks';
import type { BallObservation, Session } from './session';
import { formatVideoTime } from './videoFrames';
import { ShotStepper } from './ShotStepper';
import { movementReview, type inspectPlayerView } from './movementReview';
import { MovementPreview } from './MovementPreview';

export type CoachMode = 'auto' | 'movement' | 'match';

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
};

export function AutomaticCoach({ video, videoRef, time, onSeek, observations, fps, tracks, view, diagnostics, onAnalyze, onCancel, disabled, status, ballUnavailable, mode, onMode }: Props) {
  const movementFirst = mode === 'movement' || (mode === 'auto' && (view?.onePlayer || ballUnavailable));
  const report = useMemo(() => video && observations.length && !movementFirst
    ? automaticRecovery(observations, fps, tracks, video, diagnostics) : null, [video, observations, fps, tracks, diagnostics, movementFirst]);
  const supportedMatch = report?.players.some(player => player.examples.length) ?? false;
  const movementOnly = !!tracks && (movementFirst || (mode === 'auto' && !!report && !supportedMatch));
  const unsupportedMatch = mode === 'match' && !!tracks && !supportedMatch;
  const showShots = !!report && !movementOnly && !unsupportedMatch;
  const movement = useMemo(() => movementOnly && video && tracks && view?.trackId != null
    ? movementReview(tracks, view.trackId, video, diagnostics) : null, [movementOnly, video, tracks, view, diagnostics]);

  function jump(time: number) {
    const video = videoRef.current;
    if (!video) return;
    video.pause();
    video.currentTime = time;
    onSeek(video.currentTime);
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
      selected={mode === 'movement'} onePlayer={view?.onePlayer ?? false} onJump={jump} />}
    {unsupportedMatch && <p>Match analysis needs a supported sequence between two players. Use Single-player movement for a one-sided clip.</p>}
    {showShots && <ShotStepper steps={report.turns} time={time} readTime={() => videoRef.current?.currentTime ?? time} onJump={jump} />}
    {showShots && !tracks && <p>Import matching player tracks in Advanced tools to show estimated hitters and movement.</p>}
    {showShots && tracks && <>
      <p>Movement is the on-screen shift of the player’s box in the last {recoveryMotionSeconds * 1000} ms before the estimated contact. Check the examples against the video.</p>
      {!!report.unassignedContacts && <p>{report.unassignedContacts} contact estimates without a clear hitter.</p>}
      {report.players.map(player => {
        const measured = player.examples.flatMap(example => example.boxWidths === null ? []
          : [{ ...example, boxWidths: example.boxWidths }]).sort((a, b) => b.boxWidths - a.boxWidths);
        return <div className="recovery-card" key={player.trackId}>
          <strong>Track {player.trackId}</strong>
          <p>{measured.length} of {player.contacts} attributed contact estimates with measurable movement.
            {' '}{player.contacts - player.examples.length} lacked a supported next contact.
            {' '}{player.examples.length - measured.length} lacked continuous player tracking.</p>
          {measured.length > 0 && <ol className="rally-list" aria-label={`Movement examples for track ${player.trackId}`}>
            {measured.slice(0, 3).map(example => <li key={example.contact}>
              <div className="workspace-tools">
                <button onClick={() => jump(example.contact)}>Shot · {formatVideoTime(example.contact)}</button>
                <button onClick={() => jump(example.nextContact)}>Next contact · {formatVideoTime(example.nextContact)}</button>
              </div>
              <p>{example.boxWidths.toFixed(2)} box widths</p>
            </li>)}
          </ol>}
        </div>;
      })}
    </>}
  </section>;
}
