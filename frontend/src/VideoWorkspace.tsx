import { useEffect, useRef, useState, type RefObject } from 'react';
import { homography, imageRect, normalizePoint, TABLE, transform, validateCorners, type Point } from './geometry';
import { parseSession, type Observation, type Session, type Shot, type Player } from './session';
import { BallTrajectory } from './BallTrajectory';
import { parseDetections } from './ballDetections';

const labels = ['1 · Near-left (origin)', '2 · Near-right (width)', '3 · Far-right', '4 · Far-left'];
type Props = { url: string; name: string; size: number; videoRef: RefObject<HTMLVideoElement | null>; shots: Shot[]; onShots: (shots: Shot[]) => void; onTime: (time: number) => void; players: Player[]; selectedPlayerId: string; onPlayers: (players: Player[]) => void; onSelectPlayer: (id: string) => void; locatingPlayerId: string | null; onLocate: (id: string | null) => void };

export function VideoWorkspace({ url, name, size, videoRef, shots, onShots, onTime, players, selectedPlayerId, onPlayers, onSelectPlayer, locatingPlayerId, onLocate }: Props) {
  const [fps, setFps] = useState(25);
  const [metadata, setMetadata] = useState<Session['video'] | null>(null);
  const [calibration, setCalibration] = useState<Session['calibration']>(null);
  const [draft, setDraft] = useState<Point[] | null>(null);
  const [inspect, setInspect] = useState(false);
  const [selected, setSelected] = useState<Point | null>(null);
  const [message, setMessage] = useState('');
  const [time, setTime] = useState(0);
  const [ballObservations, setBallObservations] = useState<Session['ballObservations']>([]);
  const [rect, setRect] = useState({ x: 0, y: 0, width: 1, height: 1 });
  const importRef = useRef<HTMLInputElement>(null);
  const detectionRef = useRef<HTMLInputElement>(null);
  const [ballFps,setBallFps] = useState(25);
  const [threshold,setThreshold] = useState(.5);
  const [showBall,setShowBall] = useState(true);
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
  function seek(direction: number) {
    const video = videoRef.current;
    if (!video || !metadata) return;
    video.pause();
    video.currentTime = Math.max(0, Math.min(metadata.duration, (Math.round(video.currentTime*fps)+direction)/fps));
    onTime(video.currentTime);
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
        if (!metadata || (Object.keys(metadata) as (keyof typeof metadata)[]).some(key => session.video[key] !== metadata[key])) throw new Error('Load the matching video first (name, size, dimensions and duration must match).');
      }
      setFps(session.fps); setCalibration(session.calibration); onShots([...session.shots].sort((a,b) => a.time-b.time)); setBallObservations([...session.ballObservations].sort((a,b) => a.timestamp-b.timestamp));
      setDraft(null); setInspect(false); setSelected(null); setMessage('Session imported.');
      onPlayers(session.players); onSelectPlayer(session.selectedPlayerId); onLocate(null);
      setBallFps(session.ballFps ?? session.fps);
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
          const update = () => { const r = imageRect(v.clientWidth, v.clientHeight, v.videoWidth, v.videoHeight); setRect({x:r.x/v.clientWidth,y:r.y/v.clientHeight,width:r.width/v.clientWidth,height:r.height/v.clientHeight}); };
          update();
        }} onError={() => setMessage('Video cannot be played. Try a compatible MP4 file.')} />
        <svg className={`calibration-overlay ${draft !== null || inspect || locatingPlayerId ? 'interactive' : ''}`} viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Table calibration overlay" onPointerDown={e => {
          const video = videoRef.current;
          if (!video || !metadata || (draft === null && !inspect && !locatingPlayerId)) return;
          const bounds = e.currentTarget.getBoundingClientRect();
          const actual = imageRect(bounds.width, bounds.height, metadata.width, metadata.height);
          const point = normalizePoint({ x: e.clientX-bounds.left, y: e.clientY-bounds.top }, actual);
          if (!point) return;
          video.pause();
          if (locatingPlayerId) {
            onPlayers(players.map(p => p.id === locatingPlayerId ? { ...p, association: { timestamp: video.currentTime, position: { value: point, source: 'manual' } } } : p));
            onLocate(null); setDraft(null); setInspect(false); setMessage('Player location saved at this timestamp.');
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
          <svg x={rect.x} y={rect.y} width={rect.width} height={rect.height} viewBox="0 0 1 1" preserveAspectRatio="none">
            {showBall && <BallTrajectory videoRef={videoRef} observations={ballObservations} fps={ballFps} threshold={threshold} />}
            {players.map((p,i) => p.association && Math.abs(p.association.timestamp-time) < .5/fps && <g key={p.id}><ellipse cx={p.association.position.value.x} cy={p.association.position.value.y} rx=".012" ry=".018" /><text x={p.association.position.value.x+.015} y={p.association.position.value.y} fontSize=".035">P{i+1}: {p.name}</text></g>)}
            {corners.length > 1 && <polyline points={corners.map(p => `${p.x},${p.y}`).join(' ')+(corners.length === 4 ? ` ${corners[0].x},${corners[0].y}` : '')} />}
            {inverse && draft === null && [[{x:0,y:1.37},{x:1.525,y:1.37}],[{x:.7625,y:0},{x:.7625,y:2.74}]].map((line,i) => <polyline key={i} className="table-grid-line" points={line.map(p => transform(inverse,p)).map(p => `${p.x},${p.y}`).join(' ')} />)}
            {corners.map((p,i) => <g key={i}><ellipse cx={p.x} cy={p.y} rx=".009" ry=".014" /><text x={p.x+.012} y={p.y-.014} fontSize=".035">{i+1}</text></g>)}
          </svg>
        </svg>
      </div>
    </div>
    <div className="calibration-controls">
      <div className="workspace-tools"><button disabled={!metadata} onClick={() => detectionRef.current?.click()}>Import ball detections</button><label><input type="checkbox" checked={showBall} onChange={e => setShowBall(e.target.checked)} /> Ball overlay</label><label>Min score <input aria-label="Minimum ball score" type="number" min="0" max="1" step=".05" value={threshold} onChange={e => { const n=Number(e.target.value); if (n>=0 && n<=1) setThreshold(n); }} /></label></div>
      <input ref={detectionRef} className="visually-hidden" type="file" accept=".json,application/json" onChange={async e => {
        const file=e.target.files?.[0]; e.target.value=''; if (!file || !metadata) return;
        try { const data=parseDetections(await file.text(),metadata); setBallObservations(data.observations); setBallFps(data.fps); setMessage(`Imported ${data.observations.length} raw detections. Gaps remain empty; scores are not calibrated probabilities.`); } catch (error) { setMessage((error as Error).message); }
      }} />
      <p>{ballObservations.length} ball observations. Overlay follows decoded video frames; no interpolation across missing frames. Export JSON also saves these observations.</p>
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
