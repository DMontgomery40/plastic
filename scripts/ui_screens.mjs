// Screenshots only. Usage: node scripts/ui_screens.mjs <baseUrl> <outDir> <prefix> [session1,session2,...]
// Requires a playwright install and a chromium; PLAYWRIGHT_MODULE and CHROME_PATH can override the defaults.
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || '/workspace/om-tools/node_modules/playwright');
const [base, out, prefix, list] = process.argv.slice(2);
const sessions = (list || 'demo_locked').split(',');
const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome', args: ['--no-sandbox'] });
const shoot = async (name, session, tab, vp = { width: 1440, height: 900 }, full = true, after) => {
  const ctx = await browser.newContext({ viewport: vp, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  // the page links Google Fonts; the box may be offline, so block them and render with the fallback stack
  await page.route(/fonts\.(googleapis|gstatic)\.com/, (r) => r.abort());
  await page.goto(base, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('header');
  // pick the session: the header shows a Session select when the model has several
  const sel = page.getByRole('combobox', { name: 'Session' });
  await sel.waitFor({ timeout: 120000 }); // sessions are loaded
  await sel.selectOption(session);
  await page.waitForFunction(() => !/Loading|No text session selected/.test(document.body.innerText), null, { timeout: 120000 });
  await page.waitForTimeout(800);
  if (tab) { await page.getByRole('button', { name: new RegExp(tab, 'i') }).first().click(); await page.waitForTimeout(800); }
  if (after) await after(page);
  if (full) {
    // grow the viewport to the document instead of fullPage, which re-lays-out and blanks the charts
    const h = await page.evaluate(() => document.documentElement.scrollHeight);
    await page.setViewportSize({ width: vp.width, height: Math.min(h, 4000) });
    await page.waitForTimeout(1500);
  }
  await page.screenshot({ path: `${out}/${prefix}_${name}.png` });
  await ctx.close();
};
for (const s of sessions) {
  await shoot(`${s}_chat`, s, 'Chat');
  await shoot(`${s}_signals`, s, 'Signals');
}
// the why drawer: open the lock's first chunk (or the first non-commit chunk) from the header timeline
if (process.env.DRAWER) {
  for (const s of sessions) {
    await shoot(`${s}_drawer`, s, 'Signals', { width: 1440, height: 1000 }, false, async (page) => {
      const cell = page.getByRole('button', { name: /^Chunk \d+: (Rollback|Scale|Project)/ }).first();
      if (await cell.count()) await cell.click();
      await page.waitForTimeout(600);
    });
  }
}
// narrow layout
if (process.env.NARROW) {
  for (const s of sessions) await shoot(`${s}_narrow`, s, 'Chat', { width: 390, height: 1400 }, false);
}
await browser.close();
