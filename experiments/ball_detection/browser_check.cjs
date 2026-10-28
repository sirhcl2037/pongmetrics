// Run with Node and PLAYWRIGHT_PATH pointing to an installed Playwright package.
const { chromium } = require(process.env.PLAYWRIGHT_PATH || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
(async () => {
  const video = process.argv[2];
  const detections = path.resolve(__dirname,'output/detections.json');
  const data = JSON.parse(fs.readFileSync(detections,'utf8'));
  const browser = await chromium.launch({headless:true,channel:'chrome'});
  const page = await browser.newPage({viewport:{width:1440,height:1200}});
  const errors=[]; page.on('pageerror',e=>errors.push(e.message));
  await page.goto(process.argv[3] || 'http://127.0.0.1:4175');
  await page.locator('input[type=file]').first().setInputFiles(video);
  await page.waitForFunction(()=>document.querySelector('video')?.readyState>=2);
  // The separate detection input leaves calibration/player/shot state intact.
  await page.getByRole('button',{name:'Import ball detections',exact:true}).click();
  const input = page.locator('input[type=file]').nth(1);
  await input.setInputFiles(detections);
  await page.getByRole('status').filter({hasText:`Imported ${data.ballObservations.length}`}).waitFor();
  async function seek(t) {
    await page.locator('video').evaluate(async (v,time)=>{v.pause();await new Promise(resolve=>{v.addEventListener('seeked',resolve,{once:true});v.currentTime=time;});},t);
    await page.waitForTimeout(150);
  }
  const known=data.ballObservations.find(o=>o.timestamp>=2);
  await seek(known.timestamp);
  const dot=page.locator('.ball-trajectory ellipse');
  await dot.waitFor();
  const x=Number(await dot.getAttribute('cx'));
  if(Math.abs(x-known.position.value.x)>1e-8)throw new Error('Dot/frame mismatch');
  await page.screenshot({path:path.resolve(__dirname,'output/browser-detection.png')});
  await seek(1.4);
  if(await dot.count())throw new Error('Missing frame was filled');
  await seek(known.timestamp);
  await page.getByLabel('Minimum ball score').fill('1');
  if(await dot.count())throw new Error('Threshold not applied');
  await page.getByLabel('Minimum ball score').fill('.5');
  await page.locator('video').evaluate(v=>v.play());
  const samples=[];
  for(let i=0;i<20;i++) { await page.waitForTimeout(70);samples.push(await dot.count()?Number(await dot.getAttribute('cx')):null); }
  if(new Set(samples.filter(x=>x!==null)).size<3)throw new Error('Playback dot did not move');
  await page.locator('video').evaluate(v=>v.pause());
  await page.setViewportSize({width:390,height:844});
  if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw new Error('Mobile overflow');
  if(errors.length)throw new Error(errors.join('\n'));
  console.log(JSON.stringify({imported:data.ballObservations.length,aligned:true,gap:true,threshold:true,movingPlayback:true,mobile:true,errors}));
  await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
