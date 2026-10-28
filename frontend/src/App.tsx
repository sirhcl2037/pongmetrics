
import { useEffect, useRef, useState, type ChangeEvent } from "react";

type Wing = "FH" | "BH";
type Stroke = "Loop" | "Drive" | "Push" | "Block" | "Other";

type Shot = {
  id: number;
  time: number;
  wing: Wing;
  stroke: Stroke;
};

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const nextId = useRef(0);

  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoName, setVideoName] = useState("");
  const [shots, setShots] = useState<Shot[]>([]);
  const [stroke, setStroke] = useState<Stroke>("Loop");
  const [currentTime, setCurrentTime] = useState(0);
  const [filter, setFilter] = useState<"All" | Wing>("All");
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  function uploadVideo(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    setVideoUrl(URL.createObjectURL(file));
    setVideoName(file.name);
    setShots([]);
    setCurrentTime(0);
    setFilter("All");
    nextId.current = 0;
    event.target.value = "";
  }

  function addShot(wing: Wing) {
    if (!videoRef.current) return;

    const time = Number(videoRef.current.currentTime.toFixed(2));

    const shot: Shot = {
      id: nextId.current++,
      time,
      wing,
      stroke
    };

    setShots(previous =>
        [...previous, shot].sort((a, b) => a.time - b.time)
    );
  }

  function removeShot(id: number) {
    setShots(previous => previous.filter(shot => shot.id !== id));
  }

  function jumpTo(time: number) {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      videoRef.current.pause();
      setCurrentTime(time);
    }
  }

  const forehands = shots.filter(s => s.wing === "FH").length;
  const backhands = shots.filter(s => s.wing === "BH").length;

  const usage = shots.length ? forehands / shots.length * 100 : 0;
  const visibleShots = shots.filter(shot => filter === "All" || shot.wing === filter);
  const timestamp = (time: number) => `${Math.floor(time / 60).toString().padStart(2, "0")}:${(time % 60).toFixed(2).padStart(5, "0")}`;

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="./" aria-label="PongMetrics home"><span className="brand-mark" aria-hidden="true">p·</span>Pong<span className="brand-light">Metrics</span></a>
        <span className="workspace-label">MATCH WORKSPACE</span>
        <span className="session-badge"><i /> Local session</span>
      </header>
      <main>
        <div className="page-heading">
          <div><p className="eyebrow">YOUR GAME, IN DETAIL</p><h1>Every shot tells a story.</h1><p className="page-description">Review your match. Tag your shots. Find your patterns.</p></div>
          <button className="upload-button" onClick={() => fileRef.current?.click()}><span aria-hidden="true">↑</span> {videoUrl ? 'Change video' : 'Upload match'}</button>
          <input ref={fileRef} className="visually-hidden" type="file" accept="video/*" onChange={uploadVideo} aria-label="Upload match video" />
        </div>
        <div className="workspace-grid">
          <section className="panel video-panel" aria-labelledby="video-heading">
            <div className="panel-heading"><div className="heading-with-dot"><i /><h2 id="video-heading">Match review</h2></div><span className="small-label">{videoUrl ? 'READY TO ANNOTATE' : 'START HERE'}</span></div>
            {videoUrl ? <div className="video-stage"><video ref={videoRef} src={videoUrl} controls onTimeUpdate={e => setCurrentTime(e.currentTarget.currentTime)} /></div> :
              <button className="upload-stage" onClick={() => fileRef.current?.click()}>
                <span className="table-illustration" aria-hidden="true"><span className="table-net" /><span className="table-ball" /></span>
                <span className="upload-title">Bring your game into focus</span>
                <span className="upload-description">Choose a match video to begin your analysis.</span>
                <span className="upload-link">Select video <span aria-hidden="true">↗</span></span>
              </button>}
            <div className="video-footer"><span className="video-filename" title={videoName}>{videoName || 'No video selected'}</span><span className="privacy-note">◇ Video stays on your device</span></div>
          </section>
          <section className="panel annotation-panel" aria-labelledby="annotation-heading">
            <div className="panel-heading"><h2 id="annotation-heading">Tag a shot</h2><span className="step-badge">01</span></div>
            <div className="annotation-body">
              <div className="timestamp"><span className="eyebrow">CURRENT POSITION</span><strong>{timestamp(currentTime)}</strong><span>Tags are saved at this timestamp</span></div>
              <div className="field-heading"><span id="stroke-label">Stroke type</span><span>SELECT ONE</span></div>
              <div className="stroke-options" role="group" aria-labelledby="stroke-label">{(['Loop', 'Drive', 'Push', 'Block', 'Other'] as const).map(type => <button key={type} aria-pressed={stroke === type} className={stroke === type ? 'stroke-option selected' : 'stroke-option'} onClick={() => setStroke(type)}>{type}</button>)}</div>
              <div className="field-heading"><span>Add to timeline</span></div>
              <div className="annotation-controls"><button className="wing-button forehand" disabled={!videoUrl} onClick={() => addShot('FH')}><span className="wing-symbol">FH</span> Forehand <span aria-hidden="true">+</span></button><button className="wing-button backhand" disabled={!videoUrl} onClick={() => addShot('BH')}><span className="wing-symbol">BH</span> Backhand <span aria-hidden="true">+</span></button></div>
              <p className="annotation-hint">{videoUrl ? 'Pause at contact, select a stroke, then tag the wing.' : 'Upload a video to start tagging shots.'}</p>
            </div>
          </section>
        </div>
        <section className="stats-grid" aria-label="Session statistics">
          <div className="stat-card"><span>Total shots</span><strong>{shots.length.toString().padStart(2, '0')}</strong><small>Tagged this session</small></div>
          <div className="stat-card"><span><i className="legend-dot fh-dot" /> Forehands</span><strong>{forehands.toString().padStart(2, '0')}</strong><small>{shots.length ? `${usage.toFixed(0)}% of all shots` : 'Waiting for your first shot'}</small></div>
          <div className="stat-card"><span><i className="legend-dot bh-dot" /> Backhands</span><strong>{backhands.toString().padStart(2, '0')}</strong><small>{shots.length ? `${(100 - usage).toFixed(0)}% of all shots` : 'Waiting for your first shot'}</small></div>
          <div className="stat-card"><span>Wing balance</span><div className="balance-labels"><strong>{usage.toFixed(0)}<em>% FH</em></strong><span>{shots.length ? (100 - usage).toFixed(0) : 0}% BH</span></div><div className={`balance-track ${shots.length ? 'has-shots' : ''}`}><span style={{ width: `${usage}%` }} /></div><small>Your shot selection at a glance</small></div>
        </section>
        <section className="panel timeline-panel" aria-labelledby="timeline-heading">
          <div className="panel-heading timeline-heading"><div className="timeline-title"><h2 id="timeline-heading">Shot timeline</h2><span className="count-badge">{shots.length}</span></div><div className="filter-options" role="group" aria-label="Filter shots by wing">{(['All', 'FH', 'BH'] as const).map(value => <button key={value} aria-pressed={filter === value} className={filter === value ? 'active' : ''} onClick={() => setFilter(value)}>{value === 'All' ? 'All shots' : value === 'FH' ? 'Forehand' : 'Backhand'}</button>)}</div></div>
          {visibleShots.length ? <div className="shot-list"><div className="shot-table-heading"><span>SHOT</span><span>TIMESTAMP</span><span>WING</span><span>STROKE</span><span /></div>{visibleShots.map(shot => <div className="shot-entry" key={shot.id}><span className="shot-number">#{(shots.indexOf(shot) + 1).toString().padStart(2, '0')}</span><button className="time-link" onClick={() => jumpTo(shot.time)} aria-label={`Jump to ${timestamp(shot.time)}`}><span aria-hidden="true">▷</span> {timestamp(shot.time)}</button><span className={`wing-tag ${shot.wing === 'FH' ? 'fh-tag' : 'bh-tag'}`}>{shot.wing === 'FH' ? 'Forehand' : 'Backhand'}</span><span className="stroke-name">{shot.stroke}</span><button className="remove-button" onClick={() => removeShot(shot.id)} aria-label={`Remove shot ${shots.indexOf(shot) + 1}`} title="Remove shot">×</button></div>)}</div> : <div className="timeline-empty"><span className="empty-icon" aria-hidden="true">≡</span><h3>{shots.length ? 'No shots in this view' : 'Your rally starts here'}</h3><p>{shots.length ? 'Try a different wing filter to see your tagged shots.' : 'Tagged shots appear here. Click a timestamp to revisit the moment.'}</p></div>}
          <div className="timeline-footer"><span>{visibleShots.length} {visibleShots.length === 1 ? 'shot' : 'shots'} shown</span><span>Annotations reset when you change videos or reload.</span></div>
        </section>
        <footer className="page-footer"><span>PONGMETRICS</span><span>A closer look at your game.</span></footer>
      </main>
    </div>
  );
}

export default App;
