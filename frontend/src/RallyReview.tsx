import { useEffect, useRef, useState, type RefObject } from 'react';
import { rallyFocusOptions, rallyNoteLimit, rallyRangeError, type RallyRange, type ReviewedRally } from './rallies';
import type { Player, Shot } from './session';
import { formatVideoTime } from './videoFrames';

type Props = {
  rallies: ReviewedRally[];
  onChange: (rallies: ReviewedRally[]) => void;
  videoRef: RefObject<HTMLVideoElement | null>;
  displayedTime: number;
  duration: number;
  players: Player[];
  shots: Shot[];
};

function stopAtRallyEnd(video: HTMLVideoElement, replay: RefObject<RallyRange | null>) {
  const range = replay.current;
  if (!range || video.seeking || video.currentTime < range.end) return;
  replay.current = null;
  video.pause();
  video.currentTime = range.end;
}

export function RallyReview({ rallies, onChange, videoRef, displayedTime, duration, players, shots }: Props) {
  const [draft, setDraft] = useState<Partial<RallyRange> & { id?: string }>({});
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [message, setMessage] = useState('');
  const replay = useRef<ReviewedRally | null>(null);
  const active = rallies.find(rally => rally.id === selectedId) ?? rallies[0];
  const range = draft.start !== undefined && draft.end !== undefined ? { start: draft.start, end: draft.end } : null;
  const rangeError = range ? rallyRangeError(range, duration, rallies, draft.id) : '';
  const editing = draft.id === undefined ? -1 : rallies.findIndex(rally => rally.id === draft.id);
  const activeShots = active ? shots.filter(shot => shot.time >= active.start && shot.time < active.end) : [];

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const progress = () => stopAtRallyEnd(video, replay);
    const pause = () => { if (video.paused) replay.current = null; };
    const seek = () => {
      if (replay.current && Math.abs(video.currentTime - replay.current.start) > .001) replay.current = null;
    };
    video.addEventListener('timeupdate', progress);
    video.addEventListener('pause', pause);
    video.addEventListener('seeking', seek);
    return () => {
      replay.current = null;
      video.removeEventListener('timeupdate', progress);
      video.removeEventListener('pause', pause);
      video.removeEventListener('seeking', seek);
    };
  }, [videoRef]);

  useEffect(() => {
    const video = videoRef.current;
    if (video) stopAtRallyEnd(video, replay);
  }, [displayedTime, videoRef]);

  useEffect(() => {
    const playing = replay.current;
    if (playing && !rallies.some(rally => rally.id === playing.id && rally.start === playing.start && rally.end === playing.end)) {
      replay.current = null;
      videoRef.current?.pause();
    }
  }, [rallies, videoRef]);

  function markBoundary(boundary: keyof RallyRange) {
    const video = videoRef.current;
    if (!video) return;
    replay.current = null;
    video.pause();
    if (video.seeking || video.readyState < 2 || Math.abs(video.currentTime - displayedTime) > .1) {
      setMessage('Wait for the frame to load, then mark it.');
      return;
    }
    setDraft(previous => ({ ...previous, [boundary]: Math.min(duration, displayedTime) }));
    setMessage('');
  }

  function saveRally() {
    if (!range || rangeError) return;
    if (draft.id && editing < 0) {
      setDraft({});
      setMessage('The rally you were editing is no longer in this session.');
      return;
    }
    const rally: ReviewedRally = editing >= 0 ? { ...rallies[editing], ...range } :
      { ...range, id: crypto.randomUUID(), winnerId: null, focus: null, note: '' };
    onChange([...rallies.filter(item => item.id !== rally.id), rally].sort((a, b) => a.start - b.start));
    setSelectedId(rally.id);
    setDraft({});
    setMessage('Rally saved in this session.');
  }

  function updateRally(changes: Partial<Pick<ReviewedRally, 'winnerId' | 'focus' | 'note'>>) {
    onChange(rallies.map(rally => rally.id === active?.id ? { ...rally, ...changes } : rally));
  }

  function jumpTo(time: number) {
    const video = videoRef.current;
    if (!video) return;
    replay.current = null;
    video.pause();
    video.currentTime = time;
  }

  async function replayRally(rally: ReviewedRally) {
    const video = videoRef.current;
    if (!video) return;
    setMessage('');
    replay.current = rally;
    video.currentTime = rally.start;
    try { await video.play(); }
    catch {
      if (replay.current !== rally) return;
      replay.current = null;
      setMessage('Playback could not start. Try Replay rally again.');
    }
  }

  const unscored = rallies.filter(rally => rally.winnerId === null).length;
  const focusCounts = rallyFocusOptions.map(option => ({
    label: option.label, count: rallies.filter(rally => rally.focus === option.value).length,
  })).filter(option => option.count > 0).sort((a, b) => b.count - a.count);
  return <section className="rally-review" aria-labelledby="rally-review-heading">
    <h3 id="rally-review-heading">Rally review</h3>
    <p>Mark the start and end of a point, then replay it and record one thing to work on.</p>
    <div className="workspace-tools">
      <button disabled={!duration} onClick={() => markBoundary('start')}>Mark start</button>
      <button disabled={!duration} onClick={() => markBoundary('end')}>Mark end</button>
      {(draft.start !== undefined || draft.end !== undefined) && <span className="rally-range" aria-live="polite">
        {editing >= 0 ? `Editing rally ${editing + 1} · ` : ''}
        {draft.start === undefined ? 'Start —' : formatVideoTime(draft.start)}
        {' – '}{draft.end === undefined ? 'End —' : formatVideoTime(draft.end)}
      </span>}
      <button disabled={!range || !!rangeError} onClick={saveRally}>{draft.id ? 'Save range' : 'Save rally'}</button>
      {(draft.start !== undefined || draft.end !== undefined) && <button onClick={() => { setDraft({}); setMessage(''); }}>Clear marks</button>}
    </div>
    {(rangeError || message) && <p role="status">{rangeError || message}</p>}
    {active && <>
      <p>{rallies.length} saved {rallies.length === 1 ? 'rally' : 'rallies'} · {players.map(player =>
        `${player.name} won ${rallies.filter(rally => rally.winnerId === player.id).length}`).join(' · ')} · {unscored} unscored</p>
      {!!focusCounts.length && <p>{focusCounts.map(option => `${option.label} ${option.count}`).join(' · ')}</p>}
      <ol className="rally-list" aria-label="Saved rallies">
        {rallies.map((rally, index) => <li key={rally.id}>
          <button aria-pressed={rally.id === active.id} onClick={() => { setSelectedId(rally.id); jumpTo(rally.start); }}>
            <strong>Rally {index + 1}</strong><span>{formatVideoTime(rally.start)} – {formatVideoTime(rally.end)}</span>
          </button>
        </li>)}
      </ol>
      <div className="workspace-tools rally-actions">
        <button onClick={() => void replayRally(active)}>Replay rally</button>
        <button onClick={() => { setDraft({ id: active.id, start: active.start, end: active.end }); setMessage(''); }}>Edit range</button>
        <button onClick={() => {
          onChange(rallies.filter(rally => rally.id !== active.id));
          if (draft.id === active.id) setDraft({});
          setMessage('Rally removed.');
        }}>Remove rally</button>
      </div>
      <div className="rally-fields">
        <label>Point winner
          <select value={active.winnerId ?? ''} onChange={event => updateRally({ winnerId: event.target.value || null })}>
            <option value="">Unscored</option>
            {players.map(player => <option key={player.id} value={player.id}>{player.name}</option>)}
          </select>
        </label>
        <label>Training focus
          <select value={active.focus ?? ''} onChange={event => updateRally({ focus: (event.target.value || null) as ReviewedRally['focus'] })}>
            <option value="">Choose a focus</option>
            {rallyFocusOptions.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>
        </label>
        <label className="rally-note">What to practice
          <textarea rows={2} maxLength={rallyNoteLimit} value={active.note} onChange={event => updateRally({ note: event.target.value })} />
        </label>
      </div>
      {!!activeShots.length && <div className="rally-shot-links">
        <span>Tagged shots</span>
        {activeShots.map(shot => <button className="time-link" key={shot.id} onClick={() => jumpTo(shot.time)}>
          {formatVideoTime(shot.time)} · {players.find(player => player.id === shot.playerId)?.name ?? 'Unassigned'} · {shot.wing} {shot.stroke}
        </button>)}
      </div>}
    </>}
  </section>;
}
