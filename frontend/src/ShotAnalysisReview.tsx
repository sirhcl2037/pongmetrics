import { useMemo, useState, type RefObject } from 'react';
import { analyzeReviewedShots } from './shotAnalysis';
import { detectShotCandidates } from './shotDetection';
import { clickedPlayerTrack, playerTrackNames, type PlayerTracks } from './playerTracks';
import type { BallObservation, Player, Session, Shot } from './session';

export function ShotAnalysisReview({shots,observations,fps,threshold,tracks,players,video,videoRef}:{shots:Shot[];observations:BallObservation[];fps:number;threshold:number;tracks:PlayerTracks|null;players:Player[];video:Session['video']|null;videoRef:RefObject<HTMLVideoElement|null>}) {
  const [start,setStart]=useState(0),[end,setEnd]=useState<number|null>(null),[fixed,setFixed]=useState(false),[confirmed,setConfirmed]=useState('');
  const finish=end??video?.duration??0;
  const valid=start>=0 && finish>start && finish<=(video?.duration??0);
  const signature=JSON.stringify({start,finish,shots:shots.filter(s=>s.source==='manual').map(s=>[s.id,s.time,s.playerId])});
  const complete=confirmed===signature;
  const report=useMemo(()=>{
    if(!complete||!valid) return null;
    const names=tracks?playerTrackNames(tracks,players):new Map<number,string>();
    const ids=new Map<string,number>();
    if(tracks) for(const player of players) {
      const a=player.association;
      const match=a?clickedPlayerTrack(tracks,a.timestamp,a.position.value):null;
      if(match && names.has(match.trackId)) ids.set(player.id,match.trackId);
    }
    const candidates=detectShotCandidates(observations,fps,threshold,tracks,names.size?new Set(names.keys()):undefined);
    return analyzeReviewedShots(shots,candidates,observations,fps,threshold,tracks,ids,{start,end:finish},fixed);
  },[complete,valid,start,finish,shots,observations,fps,threshold,tracks,players,fixed]);
  function exportReport(){
    if(!report) return;
    const url=URL.createObjectURL(new Blob([JSON.stringify({format:'pongmetrics-shot-analysis-review',version:1,video,fps,minimumBallScore:threshold,
      completeContactReview:true,fixedCameraConfirmed:fixed,players,
      inputs:{ballObservations:observations,playerTracks:tracks?.json?JSON.parse(tracks.json):null},
      shots:shots.filter(s=>s.source==='manual'&&s.time>=start&&s.time<=finish),report},null,2)],{type:'application/json'}));
    const a=document.createElement('a');a.href=url;a.download=`${video?.name}.shot-analysis-review.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  const percent=(n:number|null)=>n===null?'—':`${(n*100).toFixed(0)}%`;
  return <section className="shot-suggestions" aria-label="Shot-level analysis review">
    <h3>Shot-level analysis review</h3>
    <p>Tag every actual contact in a short interval using the existing forehand/backhand controls. Include shots the detector missed. Start with 10–20 shots. Export the session to preserve your tags.</p>
    <div className="workspace-tools">
      <label>From (s) <input type="number" min="0" step=".01" value={start} onChange={e=>setStart(Number(e.target.value))}/></label>
      <label>To (s) <input type="number" min="0" step=".01" value={finish} onChange={e=>setEnd(Number(e.target.value))}/></label>
      <label><input type="checkbox" checked={fixed} onChange={e=>setFixed(e.target.checked)}/> Continuous fixed-camera view</label>
      <label><input type="checkbox" disabled={!valid} checked={complete} onChange={e=>setConfirmed(e.target.checked?signature:'')}/> I tagged every contact in this interval</label>
    </div>
    {!valid && <p role="alert">Choose an interval within the video.</p>}
    {report && <>
      <p>{report.referenceShots} manual shots · contact recall {percent(report.contactRecall)} · precision {percent(report.contactPrecision)} · {report.missedContacts} missed contacts · {report.unmatchedSuggestions.length} unmatched suggestions. Matching tolerance: ±0.12s.</p>
      <p>Ball context: {report.ballContextShots}/{report.referenceShots}. Screen movement measurable: {report.movementMeasuredShots}/{report.referenceShots}. These are evidence checks, not a validated coaching-coverage score.</p>
      <p>Player attribution: {report.details.filter(d=>d.playerAttribution==='correct').length} correct · {report.details.filter(d=>d.playerAttribution==='incorrect').length} incorrect · {report.details.filter(d=>d.playerAttribution==='unknown').length} unknown.</p>
      <h4>Footwork & recovery · evidence preview</h4>
      <p>Measures player-box displacement over 0.3s after contact, in contact-frame box widths. Arm motion can change box centers; this does not measure foot placement, recovery quality, or physical speed.</p>
      <ol className="shot-candidate-list">{report.details.map(row=><li key={row.shotId}>
        <button onClick={()=>{const v=videoRef.current;if(v){v.pause();v.currentTime=row.time;}}}>{row.time.toFixed(2)}s</button>
        <span>{row.movement?`${row.movement.bodyWidths.toFixed(2)} box widths moved`:'Movement unavailable'} · ball samples {row.ballBefore} before / {row.ballAfter} after</span>
        <details><summary>Evidence and limitations</summary><ul>{row.limitations.map(text=><li key={text}>{text}</li>)}</ul></details>
      </li>)}</ol>
      <h4>Stroke mechanics</h4><p>Available for visual pose review. Detailed joint-angle and power feedback awaits validation.</p>
      <h4>Tactical choices</h4><p>Requires verified shot outcomes and intended targets before generating recommendations.</p>
      <button onClick={exportReport}>Export shot analysis review</button>
    </>}
  </section>;
}
