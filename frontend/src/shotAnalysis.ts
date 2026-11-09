import type { BallObservation, Shot } from './session.ts';
import type { PlayerTracks } from './playerTracks.ts';
import type { ShotCandidate } from './shotDetection.ts';

export type ReviewInterval = {start:number;end:number};
export type ShotEvidence = {shotId:number;time:number;candidateTime:number|null;timingErrorSeconds:number|null;
  ballBefore:number;ballAfter:number;ballContextAvailable:boolean;trackId:number|null;
  playerAttribution:'correct'|'incorrect'|'unknown';
  movement:{bodyWidths:number;seconds:number;sampleCoverage:number}|null;limitations:string[]};

// Ordered one-to-one assignment: maximize matches first, minimize total timing
// error second. Nearest-first greedy matching can needlessly miss a second shot.
export function matchContacts(references:Shot[], candidates:ShotCandidate[], tolerance=.12) {
  const refs=[...references].sort((a,b)=>a.time-b.time), preds=[...candidates].sort((a,b)=>a.timestamp-b.timestamp);
  const table=Array.from({length:refs.length+1},()=>Array.from({length:preds.length+1},()=>({count:0,cost:0,step:''})));
  for(let i=1;i<=refs.length;i++) for(let j=1;j<=preds.length;j++) {
    const up=table[i-1][j], left=table[i][j-1];
    let best={...(up.count>left.count || (up.count===left.count && up.cost<=left.cost)?up:left),step:up.count>left.count || (up.count===left.count && up.cost<=left.cost)?'up':'left'};
    const error=Math.abs(refs[i-1].time-preds[j-1].timestamp), diagonal=table[i-1][j-1];
    if(error<=tolerance+1e-9 && (diagonal.count+1>best.count || (diagonal.count+1===best.count && diagonal.cost+error<=best.cost))) best={count:diagonal.count+1,cost:diagonal.cost+error,step:'match'};
    table[i][j]=best;
  }
  const matches:{shot:Shot;candidate:ShotCandidate}[]=[];
  let i=refs.length,j=preds.length;
  while(i>0 && j>0) {
    const step=table[i][j].step;
    if(step==='match'){matches.push({shot:refs[i-1],candidate:preds[j-1]});i--;j--;}
    else if(step==='up') i--; else j--;
  }
  return matches.reverse();
}

function movementAfter(time:number,trackId:number,tracks:PlayerTracks,end:number) {
  // Screen-space box motion only. It is not measured footwork, meters or speed.
  if(time+.3>end+1e-6) return null;
  const first=Math.round(time*tracks.fps), last=Math.round((time+.3)*tracks.fps);
  const samples=[];
  let gap=0,maxGap=0;
  for(let frame=first;frame<=last;frame++) {
    const person=tracks.frames.get(frame)?.find(p=>p.trackId===trackId);
    if(person){samples.push({frame,box:person.box});gap=0;} else {gap++;maxGap=Math.max(maxGap,gap);}
  }
  const coverage=samples.length/(last-first+1), a=samples[0], b=samples.at(-1);
  if(!a || !b || a.frame!==first || b.frame!==last || coverage<.8 || maxGap>2) return null;
  const dx=((b.box.x+b.box.width/2)-(a.box.x+a.box.width/2))*1280;
  const dy=((b.box.y+b.box.height/2)-(a.box.y+a.box.height/2))*720;
  return {bodyWidths:Math.hypot(dx,dy)/(a.box.width*1280),seconds:(last-first)/tracks.fps,sampleCoverage:coverage};
}

export function analyzeReviewedShots(shots:Shot[], candidates:ShotCandidate[], observations:BallObservation[], fps:number, threshold:number,
  tracks:PlayerTracks|null, playerTrackIds:ReadonlyMap<string,number>, interval:ReviewInterval, fixedCamera=false) {
  if(!Number.isFinite(interval.start)||!Number.isFinite(interval.end)||interval.start<0||interval.end<=interval.start) throw Error('Choose a valid review interval.');
  const tolerance=.12;
  const refs=shots.filter(s=>s.source==='manual' && s.time>=interval.start && s.time<=interval.end).sort((a,b)=>a.time-b.time);
  // Permit boundary matches; unmatched suggestions outside the interval are not false positives.
  const scoped=candidates.filter(c=>c.timestamp>=interval.start-tolerance && c.timestamp<=interval.end+tolerance);
  const matches=matchContacts(refs,scoped,tolerance), byShot=new Map(matches.map(m=>[m.shot.id,m.candidate]));
  const matchedCandidates=new Set(matches.map(m=>m.candidate));
  const unmatched=scoped.filter(c=>!matchedCandidates.has(c) && c.timestamp>=interval.start && c.timestamp<=interval.end);
  const details:ShotEvidence[]=refs.map(shot=>{
    const candidate=byShot.get(shot.id), limitations:string[]=[];
    const near=[...new Map(observations.filter(o=>Math.abs(o.timestamp-shot.time)<=.12+1e-9 && (o.position.confidence??1)>=threshold)
      .map(o=>[Math.round(o.timestamp*fps),o])).values()];
    const before=near.filter(o=>o.timestamp<shot.time-1e-6), after=near.filter(o=>o.timestamp>shot.time+1e-6);
    const ballContextAvailable=before.length>=2 && after.length>=2;
    const trackId=shot.playerId && shot.playerIdentity?.source!=='automatic' ? playerTrackIds.get(shot.playerId)??null : null;
    const playerAttribution=trackId!==null && candidate?.trackId!==null && candidate?.trackId!==undefined ?
      trackId===candidate.trackId?'correct':'incorrect':'unknown';
    const movement=fixedCamera && trackId!==null && tracks ? movementAfter(shot.time,trackId,tracks,interval.end):null;
    if(!candidate) limitations.push('Contact detector missed this manually tagged shot.');
    if(!ballContextAvailable) limitations.push('Too few nearby ball observations on one or both sides of contact.');
    if(trackId===null) limitations.push('No manually assigned player track for this shot.');
    if(playerAttribution==='incorrect') limitations.push('Contact suggestion attributes this shot to a different player.');
    if(!fixedCamera) limitations.push('Confirm a continuous fixed-camera view before comparing screen movement.');
    else if(!movement) limitations.push('Insufficient continuous player observations in the following 0.3 seconds.');
    limitations.push('Stroke mechanics, power and tactical quality are not validated by these measurements.');
    return {shotId:shot.id,time:shot.time,candidateTime:candidate?.timestamp??null,
      timingErrorSeconds:candidate?candidate.timestamp-shot.time:null,ballBefore:before.length,ballAfter:after.length,
      ballContextAvailable,trackId,playerAttribution,movement,limitations};
  });
  const predictionCount=matches.length+unmatched.length;
  return {method:'reviewed-shot-evidence-v1',interval,timingToleranceSeconds:tolerance,referenceShots:refs.length,
    matchedContacts:matches.length,missedContacts:refs.length-matches.length,unmatchedSuggestions:unmatched.map(c=>c.timestamp),
    contactRecall:refs.length?matches.length/refs.length:null,contactPrecision:predictionCount?matches.length/predictionCount:null,
    ballContextShots:details.filter(d=>d.ballContextAvailable).length,movementMeasuredShots:details.filter(d=>d.movement!==null).length,details};
}
