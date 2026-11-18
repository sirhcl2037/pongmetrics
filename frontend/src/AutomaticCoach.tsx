import { useMemo, type RefObject } from 'react';
import { automaticRecovery, recoveryMotionSeconds } from './automaticRecovery';
import type { TrackingFrame } from './ballDetections';
import type { PlayerTracks } from './playerTracks';
import type { BallObservation, Session } from './session';
import { formatVideoTime } from './videoFrames';
import { ShotStepper } from './ShotStepper';

type Props = {
  video: Session['video'] | null;
  videoRef: RefObject<HTMLVideoElement | null>;
  time: number;
  onSeek: (time: number) => void;
  observations: BallObservation[];
  fps: number;
  tracks: PlayerTracks | null;
  diagnostics: TrackingFrame[];
  onAnalyze: (() => void) | null;
  onCancel: (() => void) | null;
  disabled: boolean;
  status: string;
};

export function AutomaticCoach({ video, videoRef, time, onSeek, observations, fps, tracks, diagnostics, onAnalyze, onCancel, disabled, status }: Props) {
  const report = useMemo(() => video && observations.length
    ? automaticRecovery(observations, fps, tracks, video, diagnostics) : null, [video, observations, fps, tracks, diagnostics]);

  function jump(time: number) {
    const video = videoRef.current;
    if (!video) return;
    video.pause();
    video.currentTime = time;
    onSeek(video.currentTime);
  }

  return <section className="rally-review" aria-labelledby="automatic-coach-heading">
    <h3 id="automatic-coach-heading">Coach preview</h3>
    <p>Analyze a short, fixed-camera singles clip to step through both players’ estimated contacts.</p>
    {onAnalyze && <div className="workspace-tools">
      <button disabled={disabled || !video} onClick={onAnalyze}>Analyze clip</button>
      {onCancel && <button onClick={onCancel}>Cancel analysis</button>}
    </div>}
    {status && <p role="status">{status}</p>}
    {!onAnalyze && !report && <p>Import matching ball detections in Advanced tools to step through contacts.</p>}
    {report && <ShotStepper steps={report.turns} time={time} readTime={() => videoRef.current?.currentTime ?? time} onJump={jump} />}
    {report && !tracks && <p>Import matching player tracks in Advanced tools to show estimated hitters and movement.</p>}
    {report && tracks && (report.players.length ? <>
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
    </> : <p>Couldn’t identify a consistent pair of players in this clip.</p>)}
  </section>;
}
