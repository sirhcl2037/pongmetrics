import { useMemo, useState, type RefObject } from 'react';
import { detectShotCandidates, shotDetectionMethod } from './shotDetection';
import { playerTrackNames, type PlayerTracks } from './playerTracks';
import type { BallObservation, Player, Session } from './session';

export function ShotSuggestions({observations,fps,threshold,tracks,players,videoRef,video}:{observations:BallObservation[];fps:number;threshold:number;tracks:PlayerTracks|null;players:Player[];videoRef:RefObject<HTMLVideoElement|null>;video:Session['video']|null}) {
  const [enabled,setEnabled] = useState(false);
  const [dismissed,setDismissed] = useState<number[]>([]);
  const suggestions = useMemo(() => {
    const assigned = tracks ? playerTrackNames(tracks,players) : new Map<number,string>();
    return enabled ? detectShotCandidates(observations,fps,threshold,tracks,assigned.size ? new Set(assigned.keys()) : undefined) : [];
  },[enabled,observations,fps,threshold,tracks,players]);
  const names = tracks ? playerTrackNames(tracks,players) : new Map<number,string>();
  const visible = suggestions.filter(s => !dismissed.includes(s.timestamp));
  function exportSuggestions() {
    const data = {format:'pongmetrics-shot-candidates',version:1,coordinateSystem:'normalized-video-image',video,fps,
      method:shotDetectionMethod,minimumBallScore:threshold,candidates:suggestions.map(s => ({...s,review:dismissed.includes(s.timestamp)?'dismissed':'unreviewed'}))};
    const url = URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
    const link = document.createElement('a'); link.href=url; link.download=`${video?.name ?? 'video'}.shot-candidates.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url),1000);
  }
  return <section className="shot-suggestions" aria-label="Experimental shot detection">
    <p>Locate both players to enable suggestions across short ball gaps. Check the video before tagging a contact. Bounces, net hits and tracking errors can trigger suggestions.</p>
    <div className="workspace-tools"><button disabled={observations.length < 5} onClick={() => setEnabled(!enabled)}>{enabled?'Hide candidates':'Find shot candidates'}</button>
      {enabled && <><span>{visible.length} candidates to review</span><button onClick={exportSuggestions}>Export candidates</button><button onClick={() => setDismissed([])}>Reset dismissals</button></>}</div>
    {enabled && <ol className="shot-candidate-list">{visible.map(s => <li key={s.timestamp}>
      <button onClick={() => {const v=videoRef.current;if(v){v.pause();v.currentTime=s.timestamp;}}}>{s.timestamp.toFixed(2)}s · {s.trackId === null?'Player unknown':names.get(s.trackId) ?? `Track ${s.trackId}`}</button>
      <span>{s.gap
        ? `Estimated during a ball gap · ${s.gap.start.toFixed(2)}–${s.gap.end.toFixed(2)}s`
        : s.evidence === 'wrist-proximity' ? 'Ball turned near an estimated wrist'
        : s.evidence === 'player-proximity' ? 'Ball turned near a player' : 'Ball turned · check the hitter'}</span>
      <button aria-label={`Dismiss candidate at ${s.timestamp.toFixed(2)} seconds`} onClick={() => setDismissed(previous => [...previous,s.timestamp])}>Dismiss</button>
    </li>)}</ol>}
  </section>;
}
