const email = "hi@dozycat.ai";

function page(title: string, body: string): Response {
  const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><meta name="description" content="BetaBuddy support and privacy information for climbers.">
<title>${title} · BetaBuddy</title>
<style>
:root{font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#18201a;background:#f6f4eb}
*{box-sizing:border-box}body{margin:0}a{color:#253b21;text-underline-offset:3px}a:focus-visible{outline:3px solid #627f17;outline-offset:4px}
.wrap{max-width:920px;margin:auto;padding:28px 24px 80px}header{display:flex;align-items:center;justify-content:space-between;gap:16px;border-bottom:1px solid #cfd2c4;padding-bottom:22px}
.brand{font-size:23px;font-weight:900;letter-spacing:-.06em;text-decoration:none}.brand span{display:inline-grid;place-items:center;width:34px;height:34px;margin-right:8px;border-radius:11px;background:#d7fb48;transform:rotate(-12deg)}
nav{display:flex;gap:20px;flex-wrap:wrap}nav a{font-weight:700}main{padding-top:45px}.eyebrow{text-transform:uppercase;letter-spacing:.2em;font-size:12px;font-weight:800;color:#5c6758}
h1{font-size:clamp(42px,8vw,76px);letter-spacing:-.065em;line-height:1.03;margin:12px 0 20px}h2{font-size:25px;letter-spacing:-.035em;margin:0 0 12px}p,li{font-size:17px;line-height:1.6}p{margin:0 0 16px}.lead{font-size:20px;max-width:720px}
.cta{display:inline-block;background:#d7fb48;border:2px solid #18201a;border-radius:18px;padding:15px 22px;text-decoration:none;font-weight:800;margin:18px 0 28px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:28px 0}.card{background:#fff;border:1px solid #d8d9cf;border-radius:22px;padding:24px}.card p:last-child{margin-bottom:0}
.policy section{border-top:1px solid #cfd2c4;padding:26px 0}.policy section:first-child{border:0}footer{border-top:1px solid #cfd2c4;margin-top:44px;padding-top:24px;font-size:14px;color:#5c6758}
@media(max-width:640px){.grid{grid-template-columns:1fr}.wrap{padding:20px 18px 60px}header{align-items:flex-start;flex-direction:column}}
</style></head><body><div class="wrap"><header><a class="brand" href="/"><span aria-hidden="true">B</span>betabuddy</a><nav aria-label="Main navigation"><a href="/">Support</a><a href="/privacy">Privacy</a><a href="mailto:${email}">Contact</a></nav></header><main>${body}</main><footer>BetaBuddy · Built for the next attempt. <a href="/privacy">Privacy policy</a> · <a href="mailto:${email}">${email}</a></footer></div></body></html>`;
  return new Response(html, { headers: { "content-type": "text/html; charset=utf-8", "cache-control": "public, max-age=300", "x-content-type-options": "nosniff", "referrer-policy": "strict-origin-when-cross-origin" } });
}

const support = `<div class="eyebrow">Help for your next attempt</div><h1>BetaBuddy support</h1>
<p class="lead">Explore an offline gym and review your own climbs on your iPhone. Gym and Logbook core features are free. Lifetime membership is an optional one-time purchase, with no subscription or ads.</p>
<a class="cta" href="mailto:${email}">Email support →</a><p>When reporting a problem, include your iPhone model, iOS and BetaBuddy versions, the steps you took, and any error message. You do not need to send a personal video in your first message. Never send a password or verification code.</p>
<div class="grid">
<section class="card"><h2>No climber detected</h2><p>Use a video with one climber, their whole body visible, good lighting, and a fixed camera. Occlusion or a distant subject can reduce detection. Videos must be 0.5–180 seconds and no larger than 500 MB.</p></section>
<section class="card"><h2>Video won’t import</h2><p>Check that the original plays in Photos. If it is stored only in iCloud, let Photos download it first. Trim the clip in Photos and try again.</p></section>
<section class="card"><h2>iPhone feels hot</h2><p>Pause repeated analyses and try again later, or use Everyday review. Analysis stops if the device reaches a serious thermal state.</p></section>
<section class="card"><h2>Delete a climb</h2><p>Open the record and choose Delete to remove the app’s video copy and analysis. Your original in Photos is unaffected. Deleting the app removes local records; there is no cloud restore.</p></section>
<section class="card"><h2>Lifetime membership</h2><p>Open Settings &amp; About to buy or restore. A successful purchase adds the permanent lifetime badge to Logbook. On another device, restore using the same Apple Account.</p></section>
<section class="card"><h2>Your privacy</h2><p>The pose model is bundled. Your videos and analyses stay on your device. Read the full <a href="/privacy">privacy policy</a>.</p></section>
</div>`;

const privacy = `<div class="eyebrow">Updated November 13, 2026</div><h1>Privacy policy</h1><p class="lead">Your climbing stays on your device.</p>
<p>BetaBuddy is a climbing video review logbook and gym route viewer that runs locally on your iPhone. Core features are free. An optional one-time lifetime membership is available; there are no subscriptions, ads, or accounts.</p>
<div class="policy"><section><h2>1. What the app handles</h2><p>Videos you choose are copied into the app for on-device pose estimation. The app stores the video copy, a thumbnail, pose keypoints, analysis results, the grade and notes you enter, and whether you marked the climb as sent. None of this is sent to the developer or to any third-party server, and the app contains no advertising or analytics SDK that collects usage.</p></section>
<section><h2>2. Model and network</h2><p>The pose model ships inside the app and runs on the device. The app does not download models and does not call any cloud AI or large language model. If you pick a video that exists only in iCloud, the system may download the original first; that is part of Apple's Photos service. App Store downloads, system diagnostics and system services follow Apple's settings and policies.</p></section>
<section><h2>3. Gym</h2><p>The gym model, routes and climbing simulation in the Gym tab ship inside the app and open on the device. That page cannot reach the network. The app does not record or send which routes you look at or which body parameters you choose. The gym content is an unofficial compilation and may not match the gym as it is today; the simulation is a model's projection, not coaching advice.</p></section>
<section><h2>4. Photos and sharing</h2><p>You decide which video to import through the system photo picker; the app does not scan your photo library. When you use the system share sheet, the text you choose is sent to the app or recipient you pick and is then covered by their privacy policy.</p></section>
<section><h2>5. Storage and deletion</h2><p>Records are stored in the app's local directory, which is excluded from system backups. You can delete the video copy and analysis of any single record. Deleting the app removes its local data. Original videos in your photo library are not affected. Copies you exported must be deleted wherever you saved them. The app has no cloud restore.</p></section>
<section><h2>6. Lifetime membership purchases</h2><p>Apple handles purchases and restores through In-App Purchase. The app uses Apple's verified transaction status to display the membership badge. It does not receive payment card details or upload videos, pose results, or training records to verify a purchase. The current app does not send purchase transactions to a developer server. If server verification is enabled later, we will update this policy before doing so.</p></section>
<section><h2>7. Changes and contact</h2><p>If the way data is handled changes, this policy will be updated with the app. Privacy and support questions: <a href="mailto:${email}">${email}</a>.</p></section></div>`;

export default {
  fetch(request: Request): Response {
    const { pathname } = new URL(request.url);
    if (pathname === "/" || pathname === "/support") return page("Support", support);
    if (pathname === "/privacy" || pathname === "/privacy/" || pathname === "/privacy/en") return page("Privacy Policy", privacy);
    if (pathname === "/health") return new Response("ok", { headers: { "content-type": "text/plain; charset=utf-8" } });
    return new Response("Not found", { status: 404, headers: { "content-type": "text/plain; charset=utf-8" } });
  },
};
