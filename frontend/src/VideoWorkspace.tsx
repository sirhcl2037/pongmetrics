import { useEffect, useRef, useState, type RefObject } from 'react';
import { homography, imageRect, normalizePoint, TABLE, transform, validateCorners, type Point } from './geometry';
import { parseSession, type Observation, type Session, type Shot, type Player } from './session';
import { BallTrajectory } from './BallTrajectory';
import { ShotSuggestions } from './ShotSuggestions';
import { ShotAnalysisReview } from './ShotAnalysisReview';
import { ballTrail, parseDetections, parseTrackingFrames, type TrackingFrame } from './ballDetections';
import { parseReviewDraft, reviewDraftKey } from './reviewDraft';
import { BallTrackingDebug } from './BallTrackingDebug';
import { PlayerTrackOverlay } from './PlayerTrackOverlay';
import { BallLabeler, type BallLabel } from './BallLabeler';
import { useVideoFrameTime } from './useVideoFrameTime';
import { clickedPlayerTrack, parsePlayerTracks, type PlayerTracks } from './playerTracks';

const labels = ['1 · Near-left (origin)', '2 · Near-right (width)', '3 · Far-right', '4 · Far-left'];
type Props = { url: string; file: File; videoRef: RefObject<HTMLVideoElement | null>; shots: Shot[]; onShots: (shots: Shot[]) => void; onTime: (time: number) => void; players: Player[]; selectedPlayerId: string; onPlayers: (players: Player[]) => void; onSelectPlayer: (id: string) => void; locatingPlayerId: string | null; onLocate: (id: string | null) => void };

export function VideoWorkspace({ url, file, videoRef, shots, onShots, onTime, players, selectedPlayerId, onPlayers, onSelectPlayer, locatingPlayerId, onLocate }: Props) {
  const { name, size } = file;
  const [savedReview] = useState(() => {
    try {
      const text = localStorage.getItem(reviewDraftKey(file));
      return text ? parseReviewDraft(text, file) : null;
    } catch { return null; }
  });
  const [storageStatus, setStorageStatus] = useState(savedReview ? 'Saved review restored.' : 'Progress saves locally in this browser.');
  const displayedTime = useVideoFrameTime(videoRef);
  const [fps, setFps] = useState(savedReview?.fps ?? 25);
  const [labeling, setLabeling] = useState(false);
  const [ballLabels, setBallLabels] = useState<BallLabel[]>(savedReview?.labels ?? []);
  const [autoAdvance, setAutoAdvance] = useState(true);
  const currentLabel = ballLabels.find(label => Math.abs(label.timestamp - displayedTime) < .000001);
  function saveBallLabel(visibility: BallLabel['visibility'], point: Point | null = null, accepted = false) {
    const video = videoRef.current;
    if (!video || !video.paused || video.seeking || video.readyState < 2 || Math.abs(video.currentTime - displayedTime) > .75 / fps) {
      setMessage('Pause and wait for the frame to load before labeling.');
      return;
    }
    const label: BallLabel = { frameIndex: Math.round(displayedTime * fps), timestamp: displayedTime,
      visibility, position: point ? { value: point, source: 'manual' } : null,
      review: { action: accepted ? 'accepted' : point ? 'corrected' : 'visibility',
        proposal: proposal?.value ?? null, confidence: proposal?.confidence } };
    setBallLabels(previous => [...previous.filter(row => Math.abs(row.timestamp - displayedTime) >= .000001), label]);
    setMessage('Ball label saved.');
    if (autoAdvance) seek(1);
  }
  const [metadata, setMetadata] = useState<Session['video'] | null>(null);
  const [calibration, setCalibration] = useState<Session['calibration']>(null);
  const [draft, setDraft] = useState<Point[] | null>(null);
  const [inspect, setInspect] = useState(false);
  const [selected, setSelected] = useState<Point | null>(null);
  const [message, setMessage] = useState('');
  const [time, setTime] = useState(0);
  const [ballObservations, setBallObservations] = useState<Session['ballObservations']>(savedReview?.observations ?? []);
  const [rect, setRect] = useState({ x: 0, y: 0, width: 1, height: 1 });
  const importRef = useRef<HTMLInputElement>(null);
  const detectionRef = useRef<HTMLInputElement>(null);
  const playerTrackRef = useRef<HTMLInputElement>(null);
  const [playerTracks, setPlayerTracks] = useState<PlayerTracks | null>(null);
  const [showPlayerTracks, setShowPlayerTracks] = useState(true);
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [trackingMode, setTrackingMode] = useState<'players'|'pose'|'rtmpose'>('players');
  const [trackingStatus, setTrackingStatus] = useState('');
  const [trackingFailed, setTrackingFailed] = useState(false);
  const [trackingAttempt, setTrackingAttempt] = useState(0);
  const localTracking = ['localhost', '127.0.0.1', '[::1]'].includes(window.location.hostname);
  useEffect(() => {
    if (!metadata || !localTracking || trackingAttempt === 0) return;
    const controller = new AbortController();
    async function generateTracks(video: Session['video']) {
      setTrackingStatus(trackingMode !== 'players' ? 'Tracking body landmarks… You can annotate while this runs.' : 'Tracking players… You can annotate while this runs.');
      setTrackingFailed(false);
      try {
        const response = await fetch(trackingMode === 'rtmpose' ? '/api/body-landmarks-rtm' : trackingMode === 'pose' ? '/api/body-landmarks' : '/api/player-tracks', {
          method: 'POST', headers: { 'Content-Type': 'application/octet-stream' },
          body: file, signal: controller.signal,
        });
        const data = await response.json();
        if (!response.ok) {
          if (controller.signal.aborted) return;
          setTrackingStatus(data.error || 'Player tracking failed.');
          setTrackingFailed(true);
          return;
        }
        // The local worker uses a temporary filename for this exact upload.
        data.video.name = file.name;
        const tracks = parsePlayerTracks(JSON.stringify(data), video);
        if (controller.signal.aborted) return;
        setPlayerTracks(tracks);
        setShowPlayerTracks(true);
        setTrackingStatus(`${trackingMode === 'rtmpose' ? 'RTMPose landmarks' : trackingMode === 'pose' ? 'Body landmarks' : 'Player tracks'} ready for ${tracks.frames.size} frames.`);
      } catch (error) {
        if (controller.signal.aborted) return;
        setTrackingStatus(error instanceof Error ? error.message : 'Player tracking failed.');
        setTrackingFailed(true);
      }
    }
    void generateTracks(metadata);
    return () => controller.abort();
  }, [file, metadata, localTracking, trackingAttempt, trackingMode]);
  const [ballFps,setBallFps] = useState(savedReview?.ballFps ?? 25);
  const proposal = ballTrail(ballObservations, displayedTime, ballFps, 0).at(-1)?.position ?? null;
  useEffect(() => {
    if (!metadata) return;
    const text = JSON.stringify({ version:1, video:metadata, fps, ballFps,
      labels:ballLabels, observations:ballObservations, time:ballLabels.at(-1)?.timestamp ?? savedReview?.time ?? 0 });
    const persist = () => {
      try { localStorage.setItem(reviewDraftKey(file), text); return true; }
      catch { return false; }
    };
    const timer = window.setTimeout(() => {
      if (!persist()) setStorageStatus('Local saving failed. Export ball labels before closing this tab.');
    }, 100);
    const flush = () => { persist(); };
    window.addEventListener('pagehide', flush);
    return () => { window.clearTimeout(timer); window.removeEventListener('pagehide', flush); flush(); };
  }, [file, metadata, fps, ballFps, ballLabels, ballObservations, savedReview]);
  const [threshold,setThreshold] = useState(.5);
  const [showBall,setShowBall] = useState(true);
  const [showBallTrail, setShowBallTrail] = useState(true);
  const [trackingFrames, setTrackingFrames] = useState<TrackingFrame[]>([]);
  const [showBallDebug, setShowBallDebug] = useState(false);
  const [ballTracking, setBallTracking] = useState(false);
  const [ballStatus, setBallStatus] = useState('');
  const ballController = useRef<AbortController | null>(null);
  function cancelBallTracking() {
    ballController.current?.abort();
    ballController.current = null;
    setBallTracking(false);
    setBallStatus('');
  }
  useEffect(() => () => ballController.current?.abort(), [file]);
  async function generateBallTracks() {
    if (!metadata || ballTracking) return;
    const controller = new AbortController();
    ballController.current = controller;
    setBallTracking(true);
    setBallStatus('Tracking ball on this computer… Longer clips can take several minutes.');
    try {
      const response = await fetch('/api/ball-tracks', { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: file, signal: controller.signal });
      const data = await response.json();
      if (controller.signal.aborted) return;
      if (!response.ok) { setBallStatus(data.error || 'Ball tracking failed.'); return; }
      data.video.name = file.name;
      const text = JSON.stringify(data);
      const parsed = parseDetections(text, metadata);
      const diagnostics = parseTrackingFrames(text);
      if (controller.signal.aborted) return;
      setBallObservations(parsed.observations); setBallFps(parsed.fps); setTrackingFrames(diagnostics);
      setFps(parsed.fps);
      setMetadata(previous => previous && parsed.duration > previous.duration ? { ...previous, duration:parsed.duration } : previous);
      setThreshold(.15);
      setBallStatus(`Ball tracking ready: ${parsed.observations.length} observations. Weak detections use nearby frames for confirmation.`);
    } catch (error) {
      if (!controller.signal.aborted) setBallStatus(error instanceof Error ? error.message : 'Ball tracking failed.');
    } finally {
      if (ballController.current === controller) { setBallTracking(false); ballController.current = null; }
    }
  }
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !metadata) return;
    const observer = new ResizeObserver(() => {
      const r = imageRect(v.clientWidth, v.clientHeight, metadata.width, metadata.height);
      setRect({ x:r.x/v.clientWidth, y:r.y/v.clientHeight, width:r.width/v.clientWidth, height:r.height/v.clientHeight });
    });
    observer.observe(v);
    return () => observer.disconnect();
  }, [metadata, videoRef]);
  const corners = draft ?? calibration?.corners.map(o => o.value) ?? [];
  const inverse = calibration ? homography(TABLE, calibration.corners.map(o => o.value)) : null;
  function updateDuration(video: HTMLVideoElement) {
    if (!Number.isFinite(video.duration) || video.duration <= 0) return;
    setMetadata(previous => previous && video.duration > previous.duration ? { ...previous, duration: video.duration } : previous);
  }
  function seek(direction: number) {
    const video = videoRef.current;
    if (!video || !metadata) return;
    video.pause();
    updateDuration(video);
    const duration = Math.max(metadata.duration, Number.isFinite(video.duration) ? video.duration : 0);
    // Seek inside the frame interval to avoid decoder rounding at its boundary.
    const frame = Math.floor(video.currentTime * fps + 1e-6);
    const targetFrame = Math.max(0, frame + direction);
    video.currentTime = Math.min(duration, (targetFrame + .5) / fps);
    onTime(video.currentTime);
  }
  function nextUnreviewed() {
    const video = videoRef.current;
    if (!video || !metadata) return;
    updateDuration(video);
    const duration = Math.max(metadata.duration, Number.isFinite(video.duration) ? video.duration : 0);
    const reviewed = new Set(ballLabels.map(label => Math.round(label.timestamp * fps)));
    const total = Math.ceil(duration * fps);
    const start = Math.round(displayedTime * fps);
    for (let step = 1; step <= total; step++) {
      const frame = (start + step) % total;
      if (!reviewed.has(frame)) {
        video.pause();
        video.currentTime = Math.min(duration, (frame+.5)/fps);
        return;
      }
    }
    video.pause();
    video.currentTime = duration;
    setMessage('Known frames reviewed. Checking the video end for additional frames; use Next unreviewed again if the estimate grows.');
  }
  async function importBallLabels(backup: File) {
    try {
      if (!metadata) return;
      const data = JSON.parse(await backup.text());
      if (data?.format !== 'pongmetrics-ball-ground-truth' || data.version !== 1 ||
          data.coordinateSystem !== 'normalized-video-image' || data.video?.width !== metadata.width ||
          data.video?.height !== metadata.height) throw new Error('Load the matching video before importing labels.');
      const restored = parseReviewDraft(JSON.stringify({ version:1, video:data.video, fps:data.fps, ballFps,
        labels:data.labels, observations:[], time:0 }), file);
      // Keep newer decisions already in this tab when a backup contains the same timestamp.
      setBallLabels(previous => [...new Map([...restored.labels, ...previous].map(label => [label.timestamp, label])).values()].sort((a,b) => a.timestamp-b.timestamp));
      setFps(restored.fps);
      setMetadata(previous => previous ? { ...previous, duration:Math.max(previous.duration, restored.video.duration) } : previous);
      const video = videoRef.current;
      if (video && restored.labels.length) { video.pause(); video.currentTime = Math.max(...restored.labels.map(label => label.timestamp)); }
      setMessage(`Restored ${restored.labels.length} labels. Existing decisions in this tab were preserved.`);
    } catch (error) { setMessage(error instanceof Error ? error.message : 'Could not import labels.'); }
  }
  function startCalibration() {
    videoRef.current?.pause();
    setDraft([]); setInspect(false); setSelected(null); setMessage('');
    onLocate(null);
  }
  async function importFile(file: File) {
    try {
      const session = parseSession(await file.text());
      if (!metadata || JSON.stringify(session.video) !== JSON.stringify(metadata)) {
        // Compare named fields, independent of JSON property order.
        if (!metadata || (Object.keys(metadata) as (keyof typeof metadata)[]).some(key => session.video[key] !== metadata[key])) {
          setMessage('Load the matching video first (name, size, dimensions and duration must match).');
          return;
        }
      }
      cancelBallTracking();
      setFps(session.fps); setCalibration(session.calibration); onShots([...session.shots].sort((a,b) => a.time-b.time)); setBallObservations([...session.ballObservations].sort((a,b) => a.timestamp-b.timestamp));
      setDraft(null); setInspect(false); setSelected(null); setMessage('Session imported.');
      onPlayers(session.players); onSelectPlayer(session.selectedPlayerId); onLocate(null);
      setBallFps(session.ballFps ?? session.fps); setTrackingFrames([]);
    } catch (error) { setMessage(error instanceof Error ? error.message : 'Could not import session.'); }
  }
  function exportFile() {
    if (!metadata) return;
    const session: Session = { version: 2, video: metadata, fps, coordinateSystem: 'normalized-video-image', calibration, shots, ballObservations, players, selectedPlayerId, ballFps };
    const href = URL.createObjectURL(new Blob([JSON.stringify(session, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a'); a.href = href; a.download = `${name}.pongmetrics.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(href), 1000);
    setMessage('Session exported. Keep this JSON with the original video.');
  }
  return <>
    <div className="video-stage">
      <div className="video-image-wrapper">
        <video ref={videoRef} src={url} controls onTimeUpdate={e => { onTime(e.currentTarget.currentTime); setTime(e.currentTarget.currentTime); }} onSeeked={e => { onTime(e.currentTarget.currentTime); setTime(e.currentTarget.currentTime); }} onLoadedMetadata={e => {
          const v = e.currentTarget;
          setMetadata({ name, size, width: v.videoWidth, height: v.videoHeight, duration: v.duration });
          if (savedReview) v.currentTime = Math.min(savedReview.time, Math.max(0, v.duration-.000001));
          const update = () => { const r = imageRect(v.clientWidth, v.clientHeight, v.videoWidth, v.videoHeight); setRect({x:r.x/v.clientWidth,y:r.y/v.clientHeight,width:r.width/v.clientWidth,height:r.height/v.clientHeight}); };
          update();
        }} onDurationChange={event => updateDuration(event.currentTarget)} onError={() => setMessage('Video cannot be played. Try a compatible MP4 file.')} />
        <svg className={`calibration-overlay ${labeling || draft !== null || inspect || locatingPlayerId ? 'interactive' : ''}`} viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Table calibration overlay" onPointerDown={e => {
          const video = videoRef.current;
          if (!video || !metadata || (!labeling && draft === null && !inspect && !locatingPlayerId)) return;
          const bounds = e.currentTarget.getBoundingClientRect();
          const actual = imageRect(bounds.width, bounds.height, metadata.width, metadata.height);
          const point = normalizePoint({ x: e.clientX-bounds.left, y: e.clientY-bounds.top }, actual);
          if (!point) return;
          if (labeling) { saveBallLabel('visible', point); return; }
          video.pause();
          if (locatingPlayerId) {
            if (!playerTracks) { setMessage('Player tracking is still loading. Try again when the boxes appear.'); return; }
            const match = clickedPlayerTrack(playerTracks, displayedTime, point);
            if (!match) { setMessage('Click inside one tracking box. If boxes overlap, choose another frame.'); return; }
            const alreadyAssigned = players.some(p => p.id !== locatingPlayerId && p.association &&
              clickedPlayerTrack(playerTracks, p.association.timestamp, p.association.position.value)?.trackId === match.trackId);
            if (alreadyAssigned) { setMessage('That track already belongs to the other player. Click the other box.'); return; }
            onPlayers(players.map(p => p.id === locatingPlayerId ? { ...p, association: { timestamp: displayedTime, position: { value: point, source: 'manual' } } } : p));
            onLocate(null); setDraft(null); setInspect(false); setMessage('Player name assigned to this track.');
          } else if (draft !== null) {
            const next = [...draft, point];
            if (next.length === 4) {
              try {
                validateCorners(next);
                setCalibration({ corners: next.map(value => ({ value, source: 'manual' } as Observation<Point>)), timestamp: video.currentTime });
                setDraft(null); setInspect(true); setMessage('Calibrated. Click the table surface to inspect meters.');
              } catch (error) { setMessage((error as Error).message); setDraft([]); }
            } else setDraft(next);
          } else if (calibration) {
            try {
              const meters = transform(homography(calibration.corners.map(o => o.value), TABLE), point);
              if (meters.x < -1e-6 || meters.x > 1.525+1e-6 || meters.y < -1e-6 || meters.y > 2.74+1e-6) { setMessage('Click inside the calibrated table.'); return; }
              setSelected(meters); setMessage('');
            } catch (error) { setMessage((error as Error).message); }
          }
        }}>
          {(labeling || locatingPlayerId) && <rect x="0" y="0" width="1" height="1" fill="transparent" pointerEvents="all" />}
          <svg x={rect.x} y={rect.y} width={rect.width} height={rect.height} viewBox="0 0 1 1" preserveAspectRatio="none">
            {labeling ? <>
              {proposal && !currentLabel && <g className="ball-trajectory"><ellipse cx={proposal.value.x} cy={proposal.value.y} rx=".007" ry=".012" /><title>Detector proposal — awaiting your review</title></g>}
              {currentLabel?.position && <ellipse cx={currentLabel.position.value.x} cy={currentLabel.position.value.y} rx=".006" ry=".01" />}
            </> : <>
            {(showPlayerTracks || locatingPlayerId !== null) && playerTracks && <PlayerTrackOverlay time={displayedTime} tracks={playerTracks} players={players} showLandmarks={showLandmarks} />}
            {showBall && <BallTrajectory videoRef={videoRef} observations={ballObservations} fps={ballFps} threshold={threshold} showTrail={showBallTrail} />}
            {showBallDebug && <BallTrackingDebug videoRef={videoRef} frames={trackingFrames} fps={ballFps} />}
            {players.map((p,i) => p.association && Math.abs(p.association.timestamp-time) < .5/fps && <g key={p.id}><ellipse cx={p.association.position.value.x} cy={p.association.position.value.y} rx=".012" ry=".018" /><text x={p.association.position.value.x+.015} y={p.association.position.value.y} fontSize=".035">P{i+1}: {p.name}</text></g>)}
            {corners.length > 1 && <polyline points={corners.map(p => `${p.x},${p.y}`).join(' ')+(corners.length === 4 ? ` ${corners[0].x},${corners[0].y}` : '')} />}
            {inverse && draft === null && [[{x:0,y:1.37},{x:1.525,y:1.37}],[{x:.7625,y:0},{x:.7625,y:2.74}]].map((line,i) => <polyline key={i} className="table-grid-line" points={line.map(p => transform(inverse,p)).map(p => `${p.x},${p.y}`).join(' ')} />)}
            {corners.map((p,i) => <g key={i}><ellipse cx={p.x} cy={p.y} rx=".009" ry=".014" /><text x={p.x+.012} y={p.y-.014} fontSize=".035">{i+1}</text></g>)}
            </>}
          </svg>
        </svg>
      </div>
    </div>
    <BallLabeler enabled={labeling} onToggle={() => { videoRef.current?.pause(); setLabeling(!labeling); setDraft(null); setInspect(false); onLocate(null); }}
      time={displayedTime} fps={fps} video={metadata} labels={ballLabels} onLabel={saveBallLabel}
      onDelete={() => setBallLabels(previous => previous.filter(label => Math.abs(label.timestamp - displayedTime) >= .000001))} onStep={seek}
      proposal={proposal} onAccept={() => { if (proposal) saveBallLabel('visible', proposal.value, true); }}
      onNextUnreviewed={nextUnreviewed} autoAdvance={autoAdvance} onAutoAdvance={setAutoAdvance}
      onGenerate={localTracking ? () => void generateBallTracks() : null} generating={ballTracking} status={[ballStatus, message].filter(Boolean).join(' ')} storageStatus={storageStatus} onImport={backup => void importBallLabels(backup)} />
    <div className="calibration-controls">
      <div className="workspace-tools"><button disabled={!metadata} onClick={() => detectionRef.current?.click()}>Import ball detections</button>{localTracking && <button disabled={!metadata || ballTracking} onClick={() => void generateBallTracks()}>Generate ball tracks</button>}{ballTracking && <button onClick={() => { ballController.current?.abort(); setBallStatus('Ball tracking canceled.'); }}>Cancel ball tracking</button>}<label><input type="checkbox" checked={showBall} onChange={e => setShowBall(e.target.checked)} /> Ball overlay</label><label><input type="checkbox" checked={showBallTrail} disabled={!showBall} onChange={e => setShowBallTrail(e.target.checked)} /> Detection trail</label><label>Min score <input aria-label="Minimum ball score" type="number" min="0" max="1" step=".05" value={threshold} onChange={e => { const n=Number(e.target.value); if (n>=0 && n<=1) setThreshold(n); }} /></label>{trackingFrames.length > 0 && <label><input type="checkbox" checked={showBallDebug} onChange={e => setShowBallDebug(e.target.checked)} /> Show predictions and search radius</label>}</div>
      {ballStatus && <p role="status">{ballStatus}</p>}
      <input ref={detectionRef} className="visually-hidden" type="file" accept=".json,application/json" onChange={async e => {
        const file=e.target.files?.[0]; e.target.value=''; if (!file || !metadata) return;
        try { const text=await file.text(); const data=parseDetections(text,metadata); const diagnostics=parseTrackingFrames(text); cancelBallTracking(); setBallObservations(data.observations); setBallFps(data.fps); setFps(data.fps); setMetadata(previous => previous && data.duration > previous.duration ? { ...previous, duration:data.duration } : previous); setTrackingFrames(diagnostics); setThreshold(.15); setMessage(`Imported ${data.observations.length} detections. Gaps remain empty; scores are not calibrated probabilities.`); } catch (error) { setMessage((error as Error).message); }
      }} />
      <p>{ballObservations.length} ball observations. The fading trail shows the past half second of detected positions, with breaks at missing or low-score frames. Export JSON also saves these observations.</p>
      <div className="workspace-tools">
        <button disabled={!metadata} onClick={() => playerTrackRef.current?.click()}>Import player tracks</button>
        <label><input type="checkbox" checked={showPlayerTracks} onChange={e => setShowPlayerTracks(e.target.checked)} /> Player overlay</label>
        <label><input type="checkbox" checked={showLandmarks} onChange={e => setShowLandmarks(e.target.checked)} /> Body landmarks</label>
        {localTracking && <button disabled={!metadata || ballTracking || trackingStatus.startsWith('Tracking')} onClick={() => {setTrackingMode('pose');setTrackingAttempt(attempt => attempt + 1);}}>Generate body landmarks</button>}
        {localTracking && <button disabled={!metadata || ballTracking || trackingStatus.startsWith('Tracking')} onClick={() => {setTrackingMode('rtmpose');setTrackingAttempt(attempt => attempt + 1);}}>Generate body landmarks · RTMPose</button>}
        {localTracking && !playerTracks && !trackingFailed && <button disabled={!metadata || ballTracking || trackingStatus.startsWith('Tracking')} onClick={() => {setTrackingMode('players');setTrackingAttempt(attempt => attempt + 1);}}>Generate player tracks</button>}
        {trackingStatus.startsWith('Tracking') && <button onClick={() => {setTrackingAttempt(0);setTrackingStatus('Player analysis canceled.');}}>Cancel player analysis</button>}
        {playerTracks?.json && <button onClick={() => {
          const href=URL.createObjectURL(new Blob([playerTracks.json!],{type:'application/json'}));
          const link=document.createElement('a');link.href=href;link.download=`${name}.player-tracks.json`;link.click();
          setTimeout(() => URL.revokeObjectURL(href),1000);
        }}>Export player tracks & landmarks</button>}
        {trackingFailed && <button onClick={() => setTrackingAttempt(attempt => attempt + 1)}>Retry player tracking</button>}
        {playerTracks && <button onClick={() => {
          const video = videoRef.current;
          if (video) { video.pause(); video.currentTime = playerTracks.firstTimestamp; }
        }}>Jump to player tracks</button>}
      </div>
      {trackingStatus && <p role="status">{trackingStatus}</p>}
      <p>Body landmarks are estimated 2D joints. Hidden or low-score joints stay hidden; blurred or occluded joints can still be misplaced at high scores. Export them separately with player tracks; import through Import player tracks.</p>
      <ShotSuggestions observations={ballObservations} fps={ballFps} threshold={threshold} tracks={playerTracks} players={players} videoRef={videoRef} video={metadata} />
      <ShotAnalysisReview shots={shots} observations={ballObservations} fps={ballFps} threshold={threshold} tracks={playerTracks} players={players} videoRef={videoRef} video={metadata} />
      <input ref={playerTrackRef} className="visually-hidden" type="file" aria-label="Player tracks file" accept=".json,application/json" onChange={async e => {
        const file = e.target.files?.[0];
        e.target.value = '';
        if (!file || !metadata) return;
        try {
          const tracks = parsePlayerTracks(await file.text(), metadata);
          setPlayerTracks(tracks);
          setShowPlayerTracks(true);
          setMessage(`Imported player tracks for ${tracks.frames.size} frames.`);
        } catch (error) { setMessage(error instanceof Error ? error.message : 'Could not import player tracks.'); }
      }} />
      {locatingPlayerId && <div className="workspace-tools"><p>Click {players.find(p => p.id === locatingPlayerId)?.name} in the video.</p><button onClick={() => onLocate(null)}>Cancel player location</button></div>}
      <div className="workspace-tools"><button disabled={!metadata} onClick={() => seek(-1)}>← Previous frame</button><button disabled={!metadata} onClick={() => seek(1)}>Next frame →</button><label>FPS <input aria-label="Video FPS" type="number" min="1" max="240" value={fps} onChange={e => { const n = Number(e.target.value); if (Number.isFinite(n) && n >= 1 && n <= 240) setFps(n); }} /></label></div>
      <p>Frame steps use 1 / FPS seconds. Match FPS to your clip; variable-rate footage may not seek to exact frames.</p>
      <div className="workspace-tools"><button disabled={!metadata} onClick={startCalibration}>{calibration ? 'Recalibrate table' : 'Calibrate table'}</button>{draft !== null && <button onClick={() => { setDraft(null); setMessage(''); }}>Cancel</button>}<button disabled={!calibration || draft !== null} aria-pressed={inspect} onClick={() => { videoRef.current?.pause(); setInspect(!inspect); }}>Inspect table {inspect ? 'on' : 'off'}</button><button disabled={!metadata} onClick={exportFile}>Export JSON</button><button disabled={!metadata} onClick={() => importRef.current?.click()}>Import JSON</button><input className="visually-hidden" ref={importRef} type="file" accept=".json,application/json" onChange={e => { const file = e.target.files?.[0]; if (file) void importFile(file); e.target.value = ''; }} /></div>
      <p>{draft !== null ? `Click ${labels[draft.length]}. Order: ${labels.join(' → ')}. Near = end closest to the camera; left/right as seen in the image.` : 'Origin: near-left. x across width (1.525 m); y toward the far end (2.74 m). Recalibrate after camera movement or cuts.'}</p>
      {selected && <output>Table plane: x = {selected.x.toFixed(4)} m, y = {selected.y.toFixed(4)} m</output>}
      <p>Planar mapping applies only to the table surface, not airborne ball positions. Inspector clicks are not ball training labels.</p>
      <p role="status">{message}</p>
    </div>
  </>;
}
