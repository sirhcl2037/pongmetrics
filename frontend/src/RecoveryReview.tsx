import { useRef, useState } from 'react';
import { parseSession, type Player, type Session, type Shot } from './session';
import type { ReviewedRally } from './rallies';
import { makeRecoveryJudgment, recoverySummary, recoveryWindows, unclearReasons, type RecoveryWindow, type RecoveryJudgment, type RecoveryOutcome } from './recovery';
import { formatVideoTime } from './videoFrames';

type Props = {
  active: ReviewedRally; rallies: ReviewedRally[]; shots: Shot[]; players: Player[];
  onChange: (rallies: ReviewedRally[]) => void;
  onJump: (rally: ReviewedRally, time: number) => void;
  onReplay: (rally: ReviewedRally, start: number, end: number) => void;
  onStep: (direction: number) => void;
};
const cue = 'After each shot, recover to a balanced ready position before your partner contacts the ball. Practice 2 sets of 10 controlled exchanges, then check the same cue in your next match.';

function Summary({ windows }: { windows: RecoveryWindow[] }) {
  const s = recoverySummary(windows);
  return <>
    <p><strong>{s.ready} ready · {s.late} late · {s.unclear} unclear · {s.unreviewed} unreviewed</strong></p>
    <p>Usable judgments: {s.usable}/{s.total} tagged shots. Late: {s.lateRate === null ? '—' : `${Math.round(s.lateRate * 100)}%`} of {s.usable} usable judgments.
      {' '}{s.missingContact} without a reliable next opponent contact (included in the counts above).</p>
    {unclearReasons.map(reason => {
      const count = windows.filter(w => w.judgment?.reason === reason.value).length;
      return count ? <p key={reason.value}>{reason.label}: {count}</p> : null;
    })}
  </>;
}

export function RecoveryReview({ active, rallies, shots, players, onChange, onJump, onReplay, onStep }: Props) {
  const [playerId, setPlayerId] = useState(players[0].id);
  const [shotId, setShotId] = useState<number | null>(null);
  const [reason, setReason] = useState<RecoveryJudgment['reason']>('feet-hidden');
  const [status, setStatus] = useState('');
  const [baseline, setBaseline] = useState<Session | null>(null);
  const [baselinePlayerId, setBaselinePlayerId] = useState('');
  const comparisonInput = useRef<HTMLInputElement>(null);
  const windows = recoveryWindows(rallies, shots, playerId);
  const current = windows.filter(w => w.rally.id === active.id);
  const selected = current.find(w => w.shot.id === shotId) ?? current[0];
  const late = windows.filter(w => w.judgment?.outcome === 'late');
  const stale = rallies.flatMap(r => (r.recovery ?? []).filter(j => j.shot.playerId === playerId &&
    !windows.some(w => w.rally.id === r.id && w.judgment === j))).length;
  const name = players.find(p => p.id === playerId)?.name ?? 'Player';

  function choose(window: RecoveryWindow) {
    setShotId(window.shot.id);
    setStatus('');
    onJump(window.rally, window.shot.time);
  }
  function judge(outcome: RecoveryOutcome) {
    if (!selected) return;
    const judgment = makeRecoveryJudgment(selected, outcome, reason);
    onChange(rallies.map(r => r.id === active.id ? { ...r,
      recovery: [...(r.recovery ?? []).filter(j => j.shot.id !== selected.shot.id), judgment] } : r));
    setStatus(`Saved ${outcome} as your manual judgment. Export JSON to keep it.`);
  }
  const automatic = selected && [selected.shot, selected.opponent].some(s => s && (s.source === 'automatic' || s.playerIdentity?.source === 'automatic'));

  return <section className="recovery-review" aria-labelledby="recovery-heading">
    <div className="recovery-heading"><h4 id="recovery-heading">Recovery review</h4><span>YOUR MANUAL JUDGMENT</span></div>
    <p>Start with 10–20 points from a fixed view with your feet visible. At the opponent’s next contact, were you balanced and ready to move for the next ball?</p>
    <label className="recovery-player">Review player <select value={playerId} onChange={e => { setPlayerId(e.target.value); setShotId(null); setStatus(''); }}>
      {players.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
    </select></label>
    {!current.length ? <p>Tag {name}’s shots and the opponent’s contacts in this rally using the shot controls. Recovery review uses those timestamps; tracking is optional.</p> : <>
      <ol className="rally-list" aria-label="Recovery shots in this rally">
        {current.map(w => <li key={w.shot.id}><button aria-pressed={w.shot.id === selected?.shot.id} onClick={() => choose(w)}>
          {formatVideoTime(w.shot.time)} · {w.shot.wing} {w.shot.stroke} · {w.judgment?.outcome ?? 'unreviewed'}
        </button></li>)}
      </ol>
      {selected && <div className="recovery-card">
        <p><strong>{name} at {formatVideoTime(selected.shot.time)}</strong>{selected.opponent ?
          ` → ${players.find(p => p.id === selected.opponent!.playerId)?.name} at ${formatVideoTime(selected.opponent.time)}` : ' → next opponent contact missing or ambiguous'}</p>
        {automatic && <p>One or both contact tags or player assignments are automatic. Check them in the video before judging. Your judgment stays manual.</p>}
        <div className="workspace-tools">
          <button disabled={!selected.opponent} onClick={() => onReplay(active, selected.shot.time, selected.opponent!.time)}>Replay recovery</button>
          <button onClick={() => onJump(active, selected.shot.time)}>My contact</button>
          <button disabled={!selected.opponent} onClick={() => onJump(active, selected.opponent!.time)}>Opponent contact</button>
          <button onClick={() => onStep(-1)}>Previous frame</button>
          <button onClick={() => onStep(1)}>Next frame</button>
        </div>
        <p>Ready = balanced and prepared at their contact. Late = visibly still recovering. Unclear = evidence is missing or you cannot tell. Use the video’s playback controls and frame steps to check.</p>
        {!selected.opponent && <p>Tag the next opponent contact before choosing ready or late. A point-ending shot may have no next contact; leave it unreviewed or mark unclear.</p>}
        <div className="workspace-tools" role="group" aria-label="Recovery judgment">
          {(['ready', 'late', 'unclear'] as const).map(outcome => <button key={outcome}
            disabled={outcome !== 'unclear' && !selected.opponent} aria-pressed={selected.judgment?.outcome === outcome}
            onClick={() => judge(outcome)}>{outcome[0].toUpperCase() + outcome.slice(1)}</button>)}
          <button disabled={!active.recovery?.some(j => j.shot.id === selected.shot.id)} onClick={() => {
            onChange(rallies.map(r => r.id === active.id ? { ...r, recovery: r.recovery?.filter(j => j.shot.id !== selected.shot.id) } : r));
            setStatus('Judgment cleared.');
          }}>Clear judgment</button>
        </div>
        <label className="recovery-player">Reason for unclear <select value={selected.opponent ? reason ?? 'feet-hidden' : 'missing-contact'} disabled={!selected.opponent} onChange={e => setReason(e.target.value as RecoveryJudgment['reason'])}>
          {unclearReasons.map(r => <option key={r.value} value={r.value}>{r.label}</option>)}
        </select></label>
        {selected.judgment?.reason && <p>Saved reason: {unclearReasons.find(r => r.value === selected.judgment?.reason)?.label}. Select a reason, then press Unclear to save it.</p>}
      </div>}
    </>}
    <p role="status">{status}</p>
    <h4>{name} · all saved rallies</h4>
    <Summary windows={windows} />
    {!!stale && <p>{stale} previous judgments excluded because their contact tags or rally range changed. Review the affected shots again.</p>}
    <p>Counts cover tagged shots in saved rallies, not all returns in the footage. Unclear and unreviewed shots are excluded from the late percentage.</p>
    {!!late.length && <div className="rally-shot-links" aria-label="Late recovery examples"><strong>Jump to late examples</strong>
      {late.map(w => <button className="time-link" key={`${w.rally.id}-${w.shot.id}`} onClick={() => choose(w)}>
        Rally {rallies.indexOf(w.rally) + 1} · {formatVideoTime(w.shot.time)}
      </button>)}
    </div>}
    {late.length >= 2 && <div className="recovery-cue"><p><strong>Recurring in your review: late recovery ({late.length} shots).</strong> Practice cue: {cue}</p>
      <div className="workspace-tools"><button onClick={() => {
        onChange(rallies.map(r => r.id === active.id ? { ...r, focus: 'footwork-recovery', note: cue } : r));
        setStatus('Practice cue saved in this rally. Export JSON to keep it.');
      }} disabled={!!active.note}>Save cue to this rally</button></div>
      {!!active.note && <p>This rally already has a practice note. You can edit it above.</p>}
    </div>}
    <details className="recovery-comparison"><summary>Compare with another session</summary>
      <p>Choose an exported session, then select the same person. Use the same readiness definition and comparable camera views. This describes your reviews; it does not establish training effects.</p>
      <div className="workspace-tools"><button onClick={() => comparisonInput.current?.click()}>Open comparison JSON</button>
        {baseline && <button onClick={() => setBaseline(null)}>Clear comparison</button>}</div>
      <input ref={comparisonInput} className="visually-hidden" type="file" accept=".json,application/json" aria-label="Comparison session JSON" onChange={async e => {
        const file = e.target.files?.[0]; e.target.value = ''; if (!file) return;
        try { const session = parseSession(await file.text()); setBaseline(session); setBaselinePlayerId(session.players[0].id); setStatus('Comparison loaded. Select the same person below.'); }
        catch (error) { setStatus(error instanceof Error ? error.message : 'Could not read comparison.'); }
      }} />
      {baseline && <><p><strong>Comparison: {baseline.video.name}</strong></p>
        <label className="recovery-player">Same person in comparison <select value={baselinePlayerId} onChange={e => setBaselinePlayerId(e.target.value)}>
          {baseline.players.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
        </select></label>
        <Summary windows={recoveryWindows(baseline.rallies ?? [], baseline.shots, baselinePlayerId)} />
      </>}
    </details>
    <p>Export JSON saves recovery judgments with the session. The comparison file is only opened for this view.</p>
  </section>;
}
