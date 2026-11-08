// Uses an installed Playwright package; inference is replaced with a saved real detector response.
const { chromium } = require(process.env.PLAYWRIGHT_PATH || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

(async () => {
  const clip = process.argv[2];
  const payload = JSON.parse(fs.readFileSync(path.join(__dirname, 'output/detections.json'), 'utf8'));
  const browser = await chromium.launch({ headless: true, channel: 'chrome' });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1200 } });
    const errors = [];
    let playerRequests = 0, ballRequests = 0;
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/api/player-tracks', route => { playerRequests++; return route.abort(); });
    await page.route('**/api/ball-tracks', route => { ballRequests++; return route.fulfill({ json: payload }); });
    await page.goto(process.argv[3] || 'http://127.0.0.1:5173/');
    await page.evaluate(() => localStorage.clear());
    async function upload() {
      await page.locator('input[type=file]').first().setInputFiles(clip);
      await page.waitForFunction(() => document.querySelector('video')?.readyState >= 2);
    }
    const draft = () => page.evaluate(() => {
      const key = Object.keys(localStorage).find(key => key.startsWith('pongmetrics-ball-review-v1:'));
      return key ? JSON.parse(localStorage.getItem(key)) : null;
    });
    await upload();
    await page.getByRole('button', { name: 'Generate review proposals', exact: true }).click();
    await page.getByRole('status').filter({ hasText: 'Ball tracking ready:' }).first().waitFor();
    await page.getByRole('button', { name: 'Ball review off', exact: true }).click();
    await page.locator('.ball-trajectory ellipse').waitFor();
    await page.keyboard.press('a');
    await page.waitForFunction(() => document.querySelector('video').currentTime > .04);
    await page.waitForTimeout(250);
    assert.equal((await draft()).labels[0].review.action, 'accepted');
    const bounds = await page.locator('.calibration-overlay').boundingBox();
    await page.mouse.click(bounds.x + bounds.width * .6, bounds.y + bounds.height * .4);
    await page.waitForTimeout(250);
    assert.equal((await draft()).labels[1].review.action, 'corrected');
    await page.keyboard.press('x');
    await page.waitForTimeout(250);
    const saved = await draft();
    assert.equal(saved.labels.length, 3);
    assert.equal(saved.labels[2].visibility, 'out-of-frame');
    assert.equal(saved.labels[2].position, null);
    assert.equal(new Set(saved.labels.map(label => label.timestamp)).size, 3);
    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Export ball labels (3)', exact: true }).click();
    const download = await downloadPromise;
    const exported = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
    assert.equal(exported.annotationMode, 'detector-assisted-review');
    assert.equal(exported.labels.length, 3);
    await page.reload();
    await upload();
    await page.getByRole('button', { name: 'Export ball labels (3)', exact: true }).waitFor();
    assert.equal((await draft()).observations.length, payload.ballObservations.length);
    assert.equal(ballRequests, 1);
    assert.equal(playerRequests, 0);
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ accepted:true, corrected:true, visibility:true, advanced:true,
      exported:true, restored:true, noExtraInference:true, mobile:true, errors }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
