import { test } from 'node:test';
import assert from 'node:assert/strict';
import { homography, transform, TABLE, validateCorners, imageRect, normalizePoint } from './geometry.ts';
import { parseSession, defaultPlayers } from './session.ts';
import { ballTrail, recentBallTrail, parseDetections, parseTrackingFrames } from './ballDetections.ts';
import { clickedPlayerTrack, landmarkNames, parsePlayerTracks, playerTrackNames, type PlayerTracks } from './playerTracks.ts';
import { detectShotCandidates, playerAxisEvidence } from './shotDetection.ts';
import { analyzeReviewedShots, matchContacts } from './shotAnalysis.ts';
import type { Shot } from './session.ts';
import { displayedFrameIndex } from './videoFrames.ts';
import { parseReviewDraft } from './reviewDraft.ts';

test('MKV detections can extend an incorrect header duration while matching the same source', () => {
  const video = {name:'clip.mkv',size:100,width:3840,height:2160,duration:11};
  const data = {format:'pongmetrics-ball-detections',version:1,coordinateSystem:'normalized-video-image',
    video:{...video,duration:15.28},fps:50,ballObservations:[{timestamp:15.26,
      position:{value:{x:.4,y:.5},source:'automatic',confidence:.8}}]};
  assert.equal(parseDetections(JSON.stringify(data),video).duration,15.28);
  assert.throws(() => parseDetections(JSON.stringify(data),{...video,size:101}));
  assert.throws(() => parseDetections(JSON.stringify({...data,video:{...data.video,name:'clip.mp4'}}),{...video,name:'clip.mp4'}));
});

test('review drafts preserve human decisions and proposals and reject a different clip', () => {
  const video = {name:'review.mp4',size:100,width:1280,height:720,duration:9};
  const label = {frameIndex:0,timestamp:0,visibility:'visible',position:{value:{x:.4,y:.5},source:'manual'},
    review:{action:'accepted',proposal:{x:.4,y:.5},confidence:.8}};
  const draft = {version:1,video,fps:25,ballFps:25,labels:[label],time:0,
    observations:[{timestamp:0,position:{value:{x:.4,y:.5},source:'automatic',confidence:.8}}]};
  assert.deepEqual(parseReviewDraft(JSON.stringify(draft),video).labels,[label]);
  assert.throws(() => parseReviewDraft(JSON.stringify(draft),{...video,size:101}));
  assert.throws(() => parseReviewDraft(JSON.stringify({...draft,labels:[label,label]}),video));
  assert.throws(() => parseReviewDraft(JSON.stringify({...draft,labels:[{...label,visibility:'occluded'}]}),video));
});

test('saved player clicks bind names to track IDs and reject ambiguous assignments', () => {
  const tracks: PlayerTracks = { fps:25, firstTimestamp:0, frames:new Map([[0, [
    {trackId:7, playerId:'near-player', box:{x:0,y:.5,width:.4,height:.5}},
    {trackId:9, playerId:'far-player', box:{x:.5,y:0,width:.4,height:.4}},
  ]]]) };
  const players = defaultPlayers();
  players[0].name = 'Alex';
  players[0].association = {timestamp:0,position:{value:{x:.6,y:.2},source:'manual'}};
  assert.equal(playerTrackNames(tracks,players).get(9),'Alex');
  assert.equal(playerTrackNames(tracks,players).has(7),false);
  players[0].name = 'Sam';
  assert.equal(playerTrackNames(tracks,players).get(9),'Sam');
  players[1].association = players[0].association;
  assert.equal(playerTrackNames(tracks,players).size,0);
  players[1].association = {timestamp:0,position:{value:{x:1,y:1},source:'manual'}};
  assert.equal(playerTrackNames(tracks,players).get(9),'Sam');
  assert.equal(playerTrackNames({...tracks,frames:new Map()},players).size,0);
});
const corners = [{x:.3,y:.2},{x:.7,y:.2},{x:.9,y:.8},{x:.1,y:.8}];
const close = (a: number,b: number) => assert.ok(Math.abs(a-b)<1e-8, `${a} != ${b}`);

test('tracking diagnostics never become observations and require ordered finite values', () => {
  const trackingFrames = [{ timestamp: .04, prediction: { x: 1.2, y: -.1 }, uncertaintyRadiusPixels: 40, reason: 'occluded' }];
  assert.deepEqual(parseTrackingFrames(JSON.stringify({trackingFrames})), trackingFrames);
  assert.throws(() => parseTrackingFrames(JSON.stringify({trackingFrames:[null]})), /Invalid tracking diagnostics/);
  assert.throws(() => parseTrackingFrames(JSON.stringify({trackingFrames:['bad']})), /Invalid tracking diagnostics/);
  assert.throws(() => parseTrackingFrames(JSON.stringify({trackingFrames:[{timestamp:0,prediction:null,uncertaintyRadiusPixels:null}]})), /Invalid tracking diagnostics/);
  assert.throws(() => parseTrackingFrames(JSON.stringify({trackingFrames:[...trackingFrames,...trackingFrames]})));
  assert.throws(() => parseTrackingFrames(JSON.stringify({trackingFrames:[{...trackingFrames[0],uncertaintyRadiusPixels:-1}]})));
  const observations = Array.from({length:10000}, (_,i) => ({timestamp:i/25, position:{value:{x:.5,y:.3},source:'automatic' as const,confidence:.3}}));
  assert.equal(ballTrail(observations,399.96,25,.15).length,6);
  assert.equal(ballTrail(observations,399.96,25,.5).length,0);
});
test('ball trail drops missing frames and uncertain detections without interpolation', () => {
  const observations = [0,.04,.12,.16].map(timestamp => ({timestamp,position:{value:{x:.5,y:.3},source:'automatic' as const,confidence:.9}}));
  assert.equal(ballTrail(observations,.08,25,.5).length,0);
  assert.equal(ballTrail(observations,.12,25,.5).length,1);
  assert.equal(ballTrail(observations,.16,25,.5).length,2);
  assert.equal(ballTrail(observations,.16,25,.95).length,0);
  const video={name:'clip.mp4',size:100,width:1280,height:720,duration:9};
  const payload={format:'pongmetrics-ball-detections',version:1,coordinateSystem:'normalized-video-image',video,fps:25,ballObservations:observations};
  assert.equal(parseDetections(JSON.stringify(payload),video).observations.length,4);
  assert.equal(parseDetections(JSON.stringify({...payload,video:{...video,duration:9.04}}),video).observations.length,4);
  assert.throws(()=>parseDetections(JSON.stringify({...payload,video:{...video,duration:10}}),video));
  assert.throws(()=>parseDetections(JSON.stringify(payload),{...video,size:101}));
  assert.throws(()=>parseDetections(JSON.stringify({...payload,ballObservations:[observations[0],observations[0]]}),video));
});
test('visible trail preserves gaps and score exclusions while the current ball disappears', () => {
  const observations = [0,.04,.12,.16,.20,.24].map(timestamp => ({ timestamp,
    position:{value:{x:timestamp,y:.3},source:'automatic' as const,confidence:timestamp === .16 ? .2 : .9} }));
  const times = (time:number) => recentBallTrail(observations,time,25,.5).map(segment => segment.map(o => o.timestamp));
  assert.deepEqual(times(.24), [[0,.04],[.12],[.20,.24]]);
  assert.deepEqual(times(.08), [[0,.04]]);
  assert.equal(ballTrail(observations,.08,25,.5).length,0);
  assert.deepEqual(times(.04), [[0,.04]]); // Seeking back never retains future observations.
  assert.deepEqual(times(.8), []);
});

test('trail duration is half a second at both 25 and 50 FPS', () => {
  for (const fps of [25,50]) {
    const observations = Array.from({length:fps*2},(_,i) => ({timestamp:i/fps,
      position:{value:{x:.2,y:.3},source:'automatic' as const,confidence:.9}}));
    const segments = recentBallTrail(observations,1,fps,.5);
    assert.equal(segments.length,1);
    assert.ok(segments[0][0].timestamp >= .5);
    assert.equal(segments[0].at(-1)?.timestamp,1);
    assert.equal(segments[0].length,Math.floor(fps*.5)+1);
  }
});

test('pose imports preserve landmark scores, missing joints, and identity; reject malformed data', () => {
  const video={name:'pose.mp4',size:100,width:1280,height:720,duration:2};
  const landmarks=landmarkNames.map(name => ({name,position:name === 'nose' ? null : {value:{x:.5,y:.3},source:'automatic',confidence:.8}}));
  const person={trackId:8,playerId:'near-player',box:{x:.2,y:.1,width:.6,height:.8},landmarks};
  const data={format:'pongmetrics-player-tracks',version:1,coordinateSystem:'normalized-video-image',video,fps:25,frames:[{timestamp:0,people:[person]}]};
  const parsed=parsePlayerTracks(JSON.stringify(data),video);
  assert.deepEqual(parsed.frames.get(0)?.[0].landmarks,landmarks);
  assert.equal(parsed.frames.get(1),undefined);
  assert.equal(parsePlayerTracks(parsed.json!,video).frames.get(0)?.[0].trackId,8);
  for (const invalid of [landmarks.slice(1),landmarks.map((j,i) => i===1?{...j,name:'wrong'}:j),landmarks.map(j => ({...j,position:{value:{x:2,y:.3},source:'automatic',confidence:.8}}))]) {
    assert.throws(() => parsePlayerTracks(JSON.stringify({...data,frames:[{timestamp:0,people:[{...person,landmarks:invalid}]}]}),video));
  }
  assert.throws(() => parsePlayerTracks(JSON.stringify(data),{...video,size:101}));
});

test('shot candidates require actual ball reversals; gaps, weak samples and straight flight are excluded', () => {
  const make = (ys:number[]) => ys.map((y,i) => ({timestamp:i/25,position:{value:{x:.5,y},source:'automatic' as const,confidence:.9}}));
  const reversing=make([.5,.4,.3,.4,.5]);
  const candidates=detectShotCandidates(reversing,25,.5);
  assert.equal(candidates.length,1);
  assert.equal(candidates[0].timestamp,.08);
  assert.equal(candidates[0].trackId,null);
  assert.equal(candidates[0].evidence,'ball-reversal'); // Can be a bounce, requires review.
  assert.equal(detectShotCandidates(make([.5,.4,.3,.2,.1]),25,.5).length,0);
  assert.equal(detectShotCandidates(reversing.map(o => ({...o,position:{...o.position,value:{x:o.position.value.y,y:.5}}})),25,.5).length,1);
  assert.equal(detectShotCandidates(make([.5,.5,.5,.5,.5]),25,.5).length,0);
  assert.equal(detectShotCandidates(reversing.map((o,i) => i===2?{...o,position:{...o.position,confidence:.1}}:o),25,.5).length,0);
  assert.equal(detectShotCandidates(reversing.map((o,i) => ({...o,timestamp:o.timestamp+(i>=2?.5:0)})),25,.5).length,0);
  const landmarks=landmarkNames.map(name => ({name,position:name==='left-wrist'?{value:{x:.5,y:.3},source:'automatic' as const,confidence:.9}:null}));
  const person={trackId:7,playerId:'far-player' as const,box:{x:.4,y:.2,width:.2,height:.2},landmarks};
  const tracks:PlayerTracks={fps:25,firstTimestamp:0,frames:new Map([[2,[person]]])};
  assert.equal(detectShotCandidates(reversing,25,.5,tracks)[0].evidence,'wrist-proximity');
  assert.equal(detectShotCandidates(reversing,25,.5,tracks)[0].trackId,7);
  assert.equal(detectShotCandidates(reversing,25,.5,tracks,new Set([99]))[0].trackId,null);
  assert.equal(detectShotCandidates(reversing,25,.5,{...tracks,frames:new Map([[1,[person]]])})[0].trackId,null);
  assert.equal(detectShotCandidates(reversing,25,.5,{...tracks,frames:new Map([[2,[person,{...person,trackId:8}]]])})[0].trackId,null);
});

test('player axis favors longitudinal reversals over cross-axis bounces in different views', () => {
  const person=(trackId:number,x:number,y:number) => ({trackId,playerId:null,box:{x,y,width:.1,height:.1}});
  const pair=[person(1,.2,.4),person(2,.7,.4)];
  const hit=playerAxisEvidence(pair,[30,0],[-30,0])!;
  const bounce=playerAxisEvidence(pair,[0,30],[0,-30])!;
  assert.ok(hit.reverses && hit.alignment>.99);
  assert.equal(bounce.reverses,false);
  assert.equal(bounce.alignment,0);
  const vertical=[person(1,.4,.2),person(2,.4,.7)];
  assert.ok(playerAxisEvidence(vertical,[0,30],[0,-30])!.reverses);
  const diagonal=[person(1,.2,.2),person(2,.7,.7)];
  assert.ok(playerAxisEvidence(diagonal,[64,36],[-64,-36])!.alignment>.99);
  assert.equal(playerAxisEvidence([...pair,person(3,.5,.1)],[30,0],[-30,0]),null);
  assert.equal(playerAxisEvidence([pair[0]],[30,0],[-30,0]),null);
  assert.equal(playerAxisEvidence([pair[0],{...pair[0],trackId:2}],[30,0],[-30,0]),null);
  const observations=[.3,.4,.5,.4,.3].map((x,i) => ({timestamp:i/25,position:{value:{x,y:.45},source:'automatic' as const,confidence:.9}}));
  const tracks:PlayerTracks={fps:25,firstTimestamp:0,frames:new Map([[2,pair]])};
  const candidate=detectShotCandidates(observations,25,.5,tracks)[0];
  assert.ok(candidate.contactScore !== null && candidate.contactScore>=.79);
  assert.ok(candidate.playerAxis?.reverses);
  assert.equal(detectShotCandidates(observations,25,.5,{...tracks,frames:new Map([[1,pair]])})[0].playerAxis,null);
});

test('shot-level matching is one-to-one and maximizes matches before minimizing timing error', () => {
  const shot=(id:number,time:number):Shot=>({id,time,wing:'FH',stroke:'Other',source:'manual',playerId:'player-1'});
  const candidates=detectShotCandidates([.5,.4,.3,.4,.5].map((y,i)=>({timestamp:i/25,position:{value:{x:.5,y},source:'automatic',confidence:.9}})),25,.5);
  const candidate=(timestamp:number)=>({...candidates[0],timestamp});
  // Greedy nearest matching could consume .10 for the first shot and miss .20.
  const pairs=matchContacts([shot(1,.09),shot(2,.20)],[candidate(0),candidate(.10)],.11);
  assert.equal(pairs.length,2);
  assert.deepEqual(pairs.map(p=>p.candidate.timestamp),[0,.10]);
  assert.equal(matchContacts([shot(1,.10),shot(2,.11)],[candidate(.10)]).length,1);
  const report=analyzeReviewedShots([shot(1,1),{...shot(2,1.5),source:'automatic'}],
    [candidate(1.05),candidate(1.06),candidate(1.5),candidate(3)],[],25,.5,null,new Map(),{start:.5,end:2});
  assert.equal(report.referenceShots,1);
  assert.equal(report.matchedContacts,1);
  assert.equal(report.contactRecall,1);
  assert.equal(report.contactPrecision,1/3);
  assert.equal(report.details[0].movement,null);
  assert.equal(report.details[0].ballContextAvailable,false);
  const wrongPlayer=analyzeReviewedShots([shot(1,1)],[{...candidate(1),trackId:9}],[],25,.5,null,new Map([['player-1',7]]),{start:.5,end:2});
  assert.equal(wrongPlayer.details[0].playerAttribution,'incorrect');
  assert.equal(analyzeReviewedShots([shot(1,1)],[{...candidate(1),trackId:7}],[],25,.5,null,new Map([['player-1',7]]),{start:.5,end:2}).details[0].playerAttribution,'correct');
  assert.equal(analyzeReviewedShots([shot(1,1)], [candidate(.95)],[],25,.5,null,new Map(),{start:1,end:2}).matchedContacts,1);
});

test('shot evidence abstains on missing or unassigned people and keeps movement separate from ball coverage', () => {
  const shot:Shot={id:1,time:1,wing:'BH',stroke:'Other',source:'manual',playerId:'player-1'};
  const observations=[.92,.96,1,1.04,1.08].map(timestamp=>({timestamp,position:{value:{x:.5,y:.3},source:'automatic' as const,confidence:.9}}));
  const frames:PlayerTracks['frames']=new Map();
  for(let frame=25;frame<=33;frame++) frames.set(frame,[{trackId:7,playerId:null,box:{x:.2+(frame-25)*.01,y:.4,width:.2,height:.4}}]);
  const tracks:PlayerTracks={fps:25,firstTimestamp:1,frames};
  const run=(people:PlayerTracks|null=tracks,ids:Map<string,number>=new Map([['player-1',7]]),fixed=true,end=2)=>
    analyzeReviewedShots([shot],[],observations,25,.5,people,ids,{start:0,end},fixed);
  const report=run();
  assert.equal(report.ballContextShots,1);
  assert.equal(report.missedContacts,1); // Missing automatic contact doesn't erase usable human-tagged evidence.
  assert.equal(report.movementMeasuredShots,1);
  assert.ok(Math.abs(report.details[0].movement!.bodyWidths-.4)<1e-8);
  assert.equal(run(null).movementMeasuredShots,0);
  assert.equal(run(tracks,new Map()).movementMeasuredShots,0);
  assert.equal(run(tracks,undefined,false).movementMeasuredShots,0);
  assert.equal(run(tracks,undefined,true,1.1).movementMeasuredShots,0);
  const gappy={...tracks,frames:new Map([...frames].filter(([f])=>f<27||f>30))};
  assert.equal(run(gappy).movementMeasuredShots,0);
  const empty=analyzeReviewedShots([],[],[],25,.5,null,new Map(),{start:0,end:1});
  assert.equal(empty.contactRecall,null);
  assert.equal(empty.contactPrecision,null);
  assert.throws(()=>analyzeReviewedShots([],[],[],25,.5,null,new Map(),{start:2,end:1}));
});

test('midpoint frame stepping never draws the following ball or player frame', () => {
  for(const fps of [25,50]) {
    const observations=[88,89,90].map(frame=>({timestamp:frame/fps,position:{value:{x:frame/100,y:.3},source:'automatic' as const,confidence:.9}}));
    for(const fraction of [0,.5,.500001,.99]) {
      const time=(88+fraction)/fps;
      assert.equal(displayedFrameIndex(time,fps),88);
      assert.equal(ballTrail(observations,time,fps,.5).at(-1)?.timestamp,88/fps);
      assert.equal(recentBallTrail(observations,time,fps,.5).flat().at(-1)?.timestamp,88/fps);
    }
    assert.equal(ballTrail(observations,89/fps,fps,.5).at(-1)?.timestamp,89/fps);
    assert.equal(ballTrail(observations.slice(1),88.75/fps,fps,.5).length,0);
  }
  const tracks:PlayerTracks={fps:25,firstTimestamp:0,frames:new Map([[88,[{trackId:7,playerId:null,box:{x:0,y:0,width:1,height:1}}]],[89,[{trackId:9,playerId:null,box:{x:0,y:0,width:1,height:1}}]]])};
  assert.equal(clickedPlayerTrack(tracks,3.54,{x:.5,y:.5})?.trackId,7);
});

test('perspective corners map to table meters and back', () => {
  validateCorners(corners);
  const h = homography(corners,TABLE), inverse = homography(TABLE,corners);
  corners.forEach((p,i) => { const mapped = transform(h,p); close(mapped.x,TABLE[i].x); close(mapped.y,TABLE[i].y); });
  for (const p of [...TABLE,{x:.7625,y:1.37},{x:.15,y:2.1}]) { const roundtrip = transform(h,transform(inverse,p)); close(roundtrip.x,p.x); close(roundtrip.y,p.y); }
});
test('known affine center and letterbox normalization are resolution independent', () => {
  const h = homography([{x:0,y:0},{x:1,y:0},{x:1,y:1},{x:0,y:1}],TABLE);
  const center = transform(h,{x:.5,y:.5}); close(center.x,.7625); close(center.y,1.37);
  for (const scale of [1,2]) { const r = imageRect(800*scale,800*scale,1280,720); assert.equal(normalizePoint({x:10,y:10},r),null); const p=normalizePoint({x:400*scale,y:400*scale},r)!; close(p.x,.5); close(p.y,.5); }
});
test('invalid, crossed and collinear corners are rejected', () => {
  assert.throws(() => validateCorners([corners[0],corners[2],corners[1],corners[3]]));
  assert.throws(() => validateCorners(Array(4).fill({x:.5,y:.5})));
  assert.throws(() => validateCorners([{x:NaN,y:0},...corners.slice(1)]));
});
test('session roundtrip preserves provenance and rejects malformed data', () => {
  const s = {version:2,video:{name:'test.mp4',size:100,width:1280,height:720,duration:9},fps:25,coordinateSystem:'normalized-video-image',calibration:{timestamp:0,corners:corners.map(value=>({value,source:'manual'}))},shots:[{id:0,time:.04,wing:'FH',stroke:'Loop',source:'manual'}],ballObservations:[{timestamp:.08,position:{value:{x:.5,y:.3},source:'automatic',confidence:.8}}]};
  const migrated = parseSession(JSON.stringify(s));
  assert.equal(migrated.shots[0].playerId,null);
  assert.deepEqual(migrated.players,defaultPlayers());
  const assigned = { ...migrated, shots: [{ ...migrated.shots[0], playerId:'player-2', playerIdentity:{value:'player-2',source:'automatic',confidence:.9} }] };
  assert.deepEqual(parseSession(JSON.stringify(assigned)),assigned);
  for(const wing of ['FH','BH'] as const) {
    const serve={...assigned,shots:[{...assigned.shots[0],wing,stroke:'Serve'}]};
    assert.deepEqual(parseSession(JSON.stringify(serve)),serve);
  }
  assert.throws(()=>parseSession(JSON.stringify({...assigned,shots:[{...assigned.shots[0],stroke:'Invalid'}]})));
  assert.throws(()=>parseSession(JSON.stringify({...assigned,selectedPlayerId:'missing'})));
  assert.throws(()=>parseSession(JSON.stringify({...assigned,shots:[{...assigned.shots[0],playerId:'missing'}]})));
  assert.throws(()=>parseSession(JSON.stringify({...s,fps:0})));
  assert.throws(()=>parseSession(JSON.stringify({...s,shots:[...s.shots,...s.shots]})));
});
