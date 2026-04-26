HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AP Commander — Accounts Payable AI Agent</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#020817;--bg2:#0a1628;--bg3:#0f1f3d;
  --glass:rgba(10,22,56,0.7);
  --border:rgba(14,165,233,0.18);--border2:rgba(14,165,233,0.35);
  --accent:#0ea5e9;--teal:#14b8a6;--purple:#8b5cf6;
  --green:#22c55e;--yellow:#eab308;--red:#ef4444;--orange:#f97316;
  --text:#f0f9ff;--dim:#94a3b8;--dimmer:#475569;
  --glow:rgba(14,165,233,0.15);--glow2:rgba(20,184,166,0.12);
  --grad:linear-gradient(135deg,#0ea5e9,#14b8a6);
  --grad2:linear-gradient(135deg,#8b5cf6,#0ea5e9);
}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;line-height:1.6;overflow-x:hidden}

/* ── GRID TEXTURE ── */
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(14,165,233,0.04) 1px,transparent 1px),linear-gradient(90deg,rgba(14,165,233,0.04) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0}

/* ── NAV ── */
nav{position:fixed;top:0;left:0;right:0;z-index:100;padding:0 40px;height:64px;display:flex;align-items:center;justify-content:space-between;transition:background .3s,border-color .3s;border-bottom:1px solid transparent}
nav.scrolled{background:rgba(2,8,23,0.85);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border-color:var(--border)}
.nav-logo{display:flex;align-items:center;gap:10px;text-decoration:none;color:var(--text)}
.nav-logo svg{width:32px;height:32px;flex-shrink:0}
.nav-logo span{font-size:17px;font-weight:800;letter-spacing:-.4px;background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.nav-links{display:flex;align-items:center;gap:8px}
.nav-links a{color:var(--dim);text-decoration:none;font-size:14px;padding:6px 14px;border-radius:8px;transition:color .2s,background .2s}
.nav-links a:hover{color:var(--text);background:rgba(14,165,233,0.08)}
.nav-links .nav-cta{background:var(--accent);color:#020817;font-weight:700;padding:7px 18px;border-radius:8px}
.nav-links .nav-cta:hover{background:#38bdf8;color:#020817}
.health-pill{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--dim);padding:4px 10px;border:1px solid var(--border);border-radius:20px;margin-left:8px}
.health-dot{width:7px;height:7px;border-radius:50%;background:var(--dimmer);transition:background .3s,box-shadow .3s}
.health-dot.ok{background:var(--green);box-shadow:0 0 8px var(--green)}
.health-dot.err{background:var(--red);box-shadow:0 0 8px var(--red)}

/* ── HERO ── */
#hero{position:relative;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:100px 24px 60px;overflow:hidden;z-index:1}
.hero-glow{position:absolute;top:30%;left:50%;transform:translate(-50%,-50%);width:700px;height:700px;background:radial-gradient(ellipse,rgba(14,165,233,0.12) 0%,rgba(20,184,166,0.07) 40%,transparent 70%);pointer-events:none;animation:pulse-glow 6s ease-in-out infinite}
@keyframes pulse-glow{0%,100%{opacity:.7;transform:translate(-50%,-50%) scale(1)}50%{opacity:1;transform:translate(-50%,-50%) scale(1.08)}}
.hero-badge{display:inline-flex;align-items:center;gap:8px;background:rgba(14,165,233,0.1);border:1px solid var(--border2);border-radius:20px;padding:6px 16px;font-size:12px;color:var(--accent);font-weight:600;letter-spacing:.5px;text-transform:uppercase;margin-bottom:28px;animation:fade-up .6s ease both}
.hero-badge span{width:6px;height:6px;background:var(--accent);border-radius:50%;animation:blink 1.4s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
h1.hero-title{font-size:clamp(36px,6vw,72px);font-weight:900;line-height:1.08;letter-spacing:-2px;margin-bottom:24px;animation:fade-up .7s .1s ease both}
h1.hero-title .grad-text{background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.hero-sub{font-size:clamp(16px,2vw,20px);color:var(--dim);max-width:580px;margin-bottom:40px;line-height:1.7;animation:fade-up .7s .2s ease both}
.hero-actions{display:flex;gap:14px;flex-wrap:wrap;justify-content:center;animation:fade-up .7s .3s ease both}
.btn{display:inline-flex;align-items:center;gap:8px;padding:13px 26px;border-radius:10px;border:none;cursor:pointer;font-size:15px;font-weight:700;text-decoration:none;transition:all .2s;letter-spacing:-.2px}
.btn-primary{background:var(--grad);color:#020817;box-shadow:0 0 24px rgba(14,165,233,0.35)}
.btn-primary:hover{transform:translateY(-2px);box-shadow:0 0 36px rgba(14,165,233,0.5)}
.btn-secondary{background:transparent;color:var(--text);border:1px solid var(--border2)}
.btn-secondary:hover{background:rgba(14,165,233,0.08);border-color:var(--accent)}
.hero-meta{margin-top:48px;display:flex;gap:28px;flex-wrap:wrap;justify-content:center;animation:fade-up .7s .4s ease both}
.hero-meta-item{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--dim)}
.hero-meta-item strong{color:var(--text)}
.hero-meta-sep{color:var(--dimmer)}
@keyframes fade-up{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}

/* ── STATS ── */
#stats{position:relative;z-index:1;padding:0 40px;margin-bottom:0}
.stats-inner{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--border);border:1px solid var(--border);border-radius:16px;overflow:hidden;max-width:900px;margin:0 auto}
.stat-card{background:var(--bg2);padding:32px 24px;text-align:center;transition:background .2s}
.stat-card:hover{background:var(--bg3)}
.stat-num{font-size:40px;font-weight:900;letter-spacing:-2px;background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1}
.stat-label{font-size:12px;color:var(--dim);text-transform:uppercase;letter-spacing:.8px;margin-top:6px;font-weight:600}

/* ── SECTION SHARED ── */
section{position:relative;z-index:1;padding:100px 40px}
.section-label{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:var(--accent);margin-bottom:12px}
.section-title{font-size:clamp(28px,4vw,44px);font-weight:900;letter-spacing:-1.5px;line-height:1.1;margin-bottom:16px}
.section-title .grad-text{background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.section-sub{font-size:17px;color:var(--dim);max-width:560px;line-height:1.7}
.section-header{margin-bottom:64px}

/* ── HOW IT WORKS ── */
#how{background:linear-gradient(180deg,var(--bg) 0%,var(--bg2) 50%,var(--bg) 100%)}
.how-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:2px;position:relative}
.how-grid::before{content:'';position:absolute;top:48px;left:calc(16.66% + 16px);right:calc(16.66% + 16px);height:2px;background:linear-gradient(90deg,var(--accent),var(--teal));opacity:.4;z-index:0}
.how-card{background:var(--glass);backdrop-filter:blur(12px);border:1px solid var(--border);border-radius:16px;padding:32px 28px;position:relative;overflow:hidden;transition:border-color .25s,transform .25s,box-shadow .25s}
.how-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.how-card:nth-child(1)::before{background:var(--accent)}
.how-card:nth-child(2)::before{background:linear-gradient(90deg,var(--accent),var(--teal))}
.how-card:nth-child(3)::before{background:var(--teal)}
.how-card:hover{border-color:var(--border2);transform:translateY(-4px);box-shadow:0 20px 40px rgba(14,165,233,0.1)}
.how-icon{width:48px;height:48px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:24px;margin-bottom:20px;border:1px solid var(--border)}
.how-icon.blue{background:rgba(14,165,233,0.12)}
.how-icon.teal{background:rgba(20,184,166,0.12)}
.how-icon.purple{background:rgba(139,92,246,0.12)}
.how-step{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--accent);margin-bottom:10px}
.how-card h3{font-size:20px;font-weight:800;margin-bottom:10px;letter-spacing:-.4px}
.how-card p{font-size:14px;color:var(--dim);line-height:1.7}
.how-tags{display:flex;flex-wrap:wrap;gap:6px;margin-top:16px}
.tag{font-size:11px;padding:3px 9px;border-radius:5px;font-weight:600;font-family:monospace;background:rgba(14,165,233,0.1);color:var(--accent);border:1px solid rgba(14,165,233,0.2)}
.tag.teal{background:rgba(20,184,166,0.1);color:var(--teal);border-color:rgba(20,184,166,0.2)}
.tag.purple{background:rgba(139,92,246,0.1);color:var(--purple);border-color:rgba(139,92,246,0.2)}

/* ── EPISODE WALKTHROUGH ── */
#demo{background:var(--bg)}
.ep-meta{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:40px;align-items:center}
.ep-pill{display:inline-flex;align-items:center;gap:6px;padding:5px 14px;border-radius:20px;font-size:12px;font-weight:600;border:1px solid var(--border);background:var(--glass);color:var(--dim);backdrop-filter:blur(8px)}
.ep-pill.primary{border-color:var(--border2);color:var(--accent);background:rgba(14,165,233,0.08)}
.ep-pill .dot{width:6px;height:6px;border-radius:50%;background:var(--accent);animation:blink 1.4s ease-in-out infinite}
.ep-layout{display:grid;grid-template-columns:2fr 3fr;gap:24px;margin-bottom:48px}
.ep-obs{display:flex;flex-direction:column;gap:14px}
.doc-card{background:var(--glass);backdrop-filter:blur(12px);border:1px solid var(--border);border-radius:12px;padding:20px;transition:border-color .2s}
.doc-card:hover{border-color:rgba(14,165,233,0.3)}
.doc-label{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--accent);margin-bottom:12px;display:flex;align-items:center;gap:6px}
.doc-label::after{content:'';flex:1;height:1px;background:var(--border)}
.inv-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;gap:12px;flex-wrap:wrap}
.inv-id{font-size:11px;color:var(--dim);font-family:monospace}
.inv-vendor{font-size:15px;font-weight:800;color:var(--text);letter-spacing:-.3px}
.inv-total{font-size:26px;font-weight:900;background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-1px;white-space:nowrap}
table.items{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px}
table.items th{color:var(--dim);font-weight:600;text-align:left;padding:3px 8px 5px 0;border-bottom:1px solid var(--border);font-size:11px;text-transform:uppercase;letter-spacing:.5px}
table.items td{padding:4px 8px 4px 0;border-bottom:1px solid rgba(14,165,233,0.06)}
table.items tr:last-child td{border-bottom:none}
.freight-ok{color:var(--dim)}
.badge{display:inline-flex;align-items:center;padding:2px 8px;border-radius:5px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.4px}
.badge-open{background:rgba(34,197,94,0.12);color:var(--green);border:1px solid rgba(34,197,94,0.3)}
.po-row{margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid var(--border)}
.po-row:last-child{margin-bottom:0;padding-bottom:0;border-bottom:none}
.po-header{display:flex;align-items:center;gap:8px;margin-bottom:6px;flex-wrap:wrap}
.po-num{font-size:13px;font-weight:700;font-family:monospace;color:var(--text)}
.po-vendor{font-size:12px;color:var(--dim)}
.grn-row{margin-bottom:8px}.grn-row:last-child{margin-bottom:0}
.grn-id{font-size:12px;font-weight:700;font-family:monospace;color:var(--teal);margin-bottom:4px}
/* Step timeline */
.ep-steps{display:flex;flex-direction:column;gap:0}
.step-card{background:var(--glass);backdrop-filter:blur(12px);border:1px solid var(--border);border-radius:12px;padding:20px 22px;transition:border-color .2s}
.step-card:hover{border-color:var(--border2)}
.step-card.terminal{border-color:rgba(34,197,94,0.3)}
.step-connector{display:flex;align-items:center;padding-left:24px;height:28px;color:var(--dimmer);font-size:18px}
.step-top{display:flex;align-items:center;gap:10px;margin-bottom:12px;flex-wrap:wrap}
.step-num{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--dimmer)}
.step-action-badge{display:inline-flex;align-items:center;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:800;font-family:monospace;letter-spacing:.3px}
.step-action-badge.query{background:rgba(14,165,233,0.15);color:var(--accent);border:1px solid rgba(14,165,233,0.3)}
.step-action-badge.escalate{background:rgba(139,92,246,0.15);color:var(--purple);border:1px solid rgba(139,92,246,0.3)}
.step-action-badge.terminal-ok{background:rgba(34,197,94,0.12);color:var(--green);border:1px solid rgba(34,197,94,0.3)}
.step-json{background:rgba(2,8,23,0.7);border:1px solid var(--border);border-radius:8px;padding:12px 14px;font-size:11px;font-family:monospace;color:var(--dim);line-height:1.6;margin-bottom:12px;white-space:pre-wrap;word-break:break-word}
.step-json .k{color:var(--accent)}.step-json .v{color:var(--text)}.step-json .n{color:var(--teal)}
.step-reveal{display:flex;align-items:flex-start;gap:10px;border-left:3px solid var(--accent);border-radius:0 8px 8px 0;padding:10px 14px;font-size:12px;color:var(--text);line-height:1.6;background:rgba(14,165,233,0.07)}
.step-reveal.mgr{border-left-color:var(--purple);background:rgba(139,92,246,0.07)}
.reveal-lbl{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;white-space:nowrap;padding-top:1px;color:var(--accent)}
.reveal-lbl.mgr{color:var(--purple)}
.step-score{display:flex;align-items:center;gap:14px;margin-top:14px;padding-top:14px;border-top:1px solid var(--border)}
.step-score-num{font-size:38px;font-weight:900;letter-spacing:-1px;background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.step-score-lbl{font-size:12px;color:var(--dim);line-height:1.6}
/* Reward breakdown */
.reward-section{background:var(--glass);backdrop-filter:blur(12px);border:1px solid var(--border);border-radius:16px;padding:32px}
.reward-title{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--accent);margin-bottom:24px;display:flex;align-items:center;gap:8px}
.reward-title::after{content:'';flex:1;height:1px;background:var(--border)}
.reward-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:24px}
.rw-row{display:flex;flex-direction:column;gap:5px}
.rw-label{display:flex;justify-content:space-between;align-items:center;font-size:12px;gap:8px}
.rw-name{color:var(--dim);font-weight:600;flex:1}
.rw-val{font-weight:800;font-family:monospace;color:var(--text);white-space:nowrap}
.rw-bar-bg{height:5px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden}
.rw-bar-fill{height:100%;border-radius:3px;background:var(--grad)}
.reward-formula{background:rgba(2,8,23,0.6);border:1px solid var(--border);border-radius:8px;padding:16px;font-size:12px;font-family:monospace;color:var(--dim);line-height:2;text-align:center}
.reward-formula .hl{color:var(--accent);font-weight:700}
.reward-formula .total{color:var(--green);font-size:16px;font-weight:900}

/* ── FOOTER ── */
footer{position:relative;z-index:1;background:var(--bg2);border-top:1px solid var(--border);padding:48px 40px 32px}
.footer-inner{max-width:1100px;margin:0 auto;display:grid;grid-template-columns:1fr auto;gap:32px;align-items:start}
.footer-brand{display:flex;align-items:center;gap:10px;margin-bottom:12px}
.footer-brand span{font-size:16px;font-weight:800;background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.footer-desc{font-size:13px;color:var(--dim);max-width:320px;line-height:1.7}
.footer-links{display:flex;flex-direction:column;gap:10px;align-items:flex-end}
.footer-links a{color:var(--dim);text-decoration:none;font-size:13px;transition:color .2s}
.footer-links a:hover{color:var(--accent)}
.footer-bottom{max-width:1100px;margin:32px auto 0;padding-top:20px;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;font-size:12px;color:var(--dimmer)}

/* ── SPINNER & MISC ── */
.spinner{display:inline-block;width:14px;height:14px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .6s linear infinite;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}
.hidden{display:none!important}
.fade-in{opacity:0;transform:translateY(24px);transition:opacity .6s,transform .6s}
.fade-in.visible{opacity:1;transform:translateY(0)}
.stagger-1{transition-delay:.1s}
.stagger-2{transition-delay:.2s}
.stagger-3{transition-delay:.3s}
.container{max-width:1100px;margin:0 auto}

@media(max-width:900px){
  .how-grid{grid-template-columns:1fr}
  .how-grid::before{display:none}
  .demo-grid{grid-template-columns:1fr}
  .action-grid{grid-template-columns:1fr}
  .stats-inner{grid-template-columns:repeat(2,1fr)}
  .footer-inner{grid-template-columns:1fr}
  .footer-links{align-items:flex-start}
  nav{padding:0 20px}
  section{padding:70px 20px}
  #stats{padding:0 20px}
}
</style>
</head>
<body>

<!-- ── NAV ── -->
<nav id="nav">
  <a class="nav-logo" href="#">
    <svg viewBox="0 0 32 32" fill="none"><path d="M16 2L28 8.5V23.5L16 30L4 23.5V8.5L16 2Z" stroke="url(#g1)" stroke-width="1.5"/><path d="M16 8L22 11.5V18.5L16 22L10 18.5V11.5L16 8Z" fill="url(#g2)" opacity=".6"/><defs><linearGradient id="g1" x1="4" y1="2" x2="28" y2="30" gradientUnits="userSpaceOnUse"><stop stop-color="#0ea5e9"/><stop offset="1" stop-color="#14b8a6"/></linearGradient><linearGradient id="g2" x1="10" y1="8" x2="22" y2="22" gradientUnits="userSpaceOnUse"><stop stop-color="#0ea5e9"/><stop offset="1" stop-color="#14b8a6"/></linearGradient></defs></svg>
    <span>AP Commander</span>
  </a>
  <div class="nav-links">
    <a href="#how">How It Works</a>
    <a href="#demo">Episode Demo</a>
    <a href="https://pathikreet-ap-clerk-env.hf.space/docs" target="_blank">API Docs ↗</a>
    <a href="https://github.com/Vayuputra2401/RL-Agent" target="_blank">GitHub ↗</a>
    <a href="#demo" class="nav-cta">See Demo</a>
    <div class="health-pill">
      <span id="health-dot" class="health-dot" title="Checking…"></span>
      <span id="health-text" style="font-size:11px;">Connecting</span>
    </div>
  </div>
</nav>

<!-- ── HERO ── -->
<section id="hero">
  <div class="hero-glow"></div>
  <div class="hero-badge"><span></span>Multi-Agent RL Environment &nbsp;·&nbsp; Always Live</div>
  <h1 class="hero-title">The AI Agent That<br><span class="grad-text">Pays Invoices</span><br>— and Catches Fraud</h1>
  <p class="hero-sub">AP Commander trains large language models to navigate enterprise Accounts Payable workflows with the rigor a CFO would require. 24 tasks. 2 AI agents. Rewards with no shortcuts.</p>
  <div class="hero-actions">
    <a href="#demo" class="btn btn-primary">▶ &nbsp;See Episode Demo</a>
    <a href="https://pathikreet-ap-clerk-env.hf.space/docs" target="_blank" class="btn btn-secondary">API Documentation ↗</a>
  </div>
  <div class="hero-meta">
    <div class="hero-meta-item"><strong>24</strong>&nbsp;Tasks</div>
    <div class="hero-meta-sep">·</div>
    <div class="hero-meta-item"><strong>2</strong>&nbsp;AI Agents</div>
    <div class="hero-meta-sep">·</div>
    <div class="hero-meta-item"><strong>16-step</strong>&nbsp;episodes</div>
    <div class="hero-meta-sep">·</div>
    <div class="hero-meta-item"><strong>5-component</strong>&nbsp;reward</div>
  </div>
</section>

<!-- ── STATS ── -->
<div id="stats" style="padding:0 40px 80px;">
  <div class="stats-inner container">
    <div class="stat-card">
      <div class="stat-num" id="stat-tasks">24</div>
      <div class="stat-label">Tasks</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" id="stat-agents">2</div>
      <div class="stat-label">AI Agents</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" id="stat-episodes">—</div>
      <div class="stat-label">Episodes Run</div>
    </div>
    <div class="stat-card">
      <div class="stat-num" id="stat-score">—</div>
      <div class="stat-label">Mean Score</div>
    </div>
  </div>
</div>

<!-- ── HOW IT WORKS ── -->
<section id="how">
  <div class="container">
    <div class="section-header" style="text-align:center;max-width:600px;margin:0 auto 64px;">
      <div class="section-label">How It Works</div>
      <h2 class="section-title">From Invoice to <span class="grad-text">Reward Signal</span></h2>
      <p class="section-sub" style="margin:0 auto;">Every episode is a complete enterprise financial scenario — generated fresh from a seeded RNG. No static dataset. The agent must reason, not memorise.</p>
    </div>
    <div class="how-grid">
      <div class="how-card fade-in">
        <div class="how-icon blue">📄</div>
        <div class="how-step">Step 01</div>
        <h3>Invoice Arrives</h3>
        <p>The agent receives a structured observation: vendor invoice, purchase orders, goods receipts, paid ledger, and company policy — all randomised per seed.</p>
        <div class="how-tags">
          <span class="tag">Invoice</span>
          <span class="tag">PO</span>
          <span class="tag">GRN</span>
          <span class="tag">Policy</span>
        </div>
      </div>
      <div class="how-card fade-in stagger-1">
        <div class="how-icon teal">🔍</div>
        <div class="how-step">Step 02</div>
        <h3>Agent Investigates</h3>
        <p>Multi-step reasoning up to 16 steps. Intermediate actions trigger simulated workplace actors — a vendor response, a manager escalation, a compliance review.</p>
        <div class="how-tags">
          <span class="tag teal">QUERY_VENDOR</span>
          <span class="tag teal">ESCALATE</span>
          <span class="tag teal">HOLD</span>
        </div>
      </div>
      <div class="how-card fade-in stagger-2">
        <div class="how-icon purple">💯</div>
        <div class="how-step">Step 03</div>
        <h3>Reward Scored</h3>
        <p>Five-component partial credit: decision accuracy, amount within 1%, reason code, explanation quality, and process bonus for the correct investigative sequence.</p>
        <div class="how-tags">
          <span class="tag purple">Partial credit</span>
          <span class="tag purple">No shortcuts</span>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ── DEMO ── -->
<section id="demo" style="background:linear-gradient(180deg,var(--bg) 0%,var(--bg2) 100%);">
  <div class="container">
    <div class="section-header">
      <div class="section-label">Episode Walkthrough</div>
      <h2 class="section-title">See the Agent <span class="grad-text">Reason</span></h2>
      <p class="section-sub">A real episode from the environment — same invoice, same policy, same reward function the model trains against. This is what a well-trained agent learns to do.</p>
    </div>

    <!-- Episode metadata -->
    <div class="ep-meta">
      <div class="ep-pill primary"><span class="dot"></span>&nbsp;long_invoice_dispute</div>
      <div class="ep-pill">Seed 42</div>
      <div class="ep-pill">max 12 steps</div>
      <div class="ep-pill">difficulty: long-horizon</div>
      <div class="ep-pill">3 actions taken</div>
    </div>

    <div class="ep-layout">
      <!-- Left: observation documents -->
      <div class="ep-obs">
        <div class="doc-card">
          <div class="doc-label">Invoice</div>
          <div class="inv-header">
            <div><div class="inv-id">INV-2024-7831</div><div class="inv-vendor">TechProcure Global</div></div>
            <div class="inv-total">$4,662.00</div>
          </div>
          <table class="items">
            <thead><tr><th>Item</th><th style="text-align:right">Qty</th><th style="text-align:right">Unit</th><th style="text-align:right">Total</th></tr></thead>
            <tbody>
              <tr><td>ThinkPad L15 Gen-4</td><td style="text-align:right">12</td><td style="text-align:right;color:var(--red);font-weight:700">$385.00</td><td style="text-align:right;font-weight:600">$4,620.00</td></tr>
              <tr class="freight-ok"><td colspan="3">Freight</td><td style="text-align:right">$42.00</td></tr>
            </tbody>
          </table>
        </div>

        <div class="doc-card">
          <div class="doc-label">Purchase Order</div>
          <div class="po-row">
            <div class="po-header">
              <span class="po-num">PO-2847</span>
              <span class="badge badge-open">OPEN</span>
              <span class="po-vendor">TechProcure Global</span>
            </div>
            <div style="font-size:11px;color:var(--dim);font-family:monospace">ThinkPad L15 Gen-4 · qty 12 @ <span style="color:var(--green);font-weight:700">$350.00</span></div>
            <div style="font-size:10px;color:var(--yellow);margin-top:5px;font-weight:600">⚠ Invoice price $385.00 exceeds agreed $350.00 by 10%</div>
          </div>
        </div>

        <div class="doc-card">
          <div class="doc-label">Goods Receipt</div>
          <div class="grn-row">
            <div class="grn-id">GRN-1094</div>
            <div style="font-size:11px;color:var(--dim);font-family:monospace">ThinkPad L15 Gen-4 · received <strong style="color:var(--teal)">12</strong></div>
          </div>
        </div>

        <div class="doc-card">
          <div class="doc-label">Company Policy</div>
          <div style="font-size:11px;color:var(--dim);font-family:monospace;line-height:1.8">
            Unit prices must match agreed PO price.<br>
            Deviation <span style="color:var(--yellow);font-weight:700">&gt;1.0%</span> must be queried<br>
            and rejected until corrected invoice received.
          </div>
        </div>
      </div>

      <!-- Right: step timeline -->
      <div class="ep-steps">

        <div class="step-card">
          <div class="step-top">
            <span class="step-num">Step 1 of 3</span>
            <span class="step-action-badge query">QUERY_VENDOR</span>
          </div>
          <div class="step-json"><span class="k">"decision":</span> <span class="v">"QUERY_VENDOR"</span>, <span class="k">"approved_amount":</span> <span class="n">0.00</span>,
<span class="k">"reason_code":</span> <span class="v">"PENDING_CLARIFICATION"</span>,
<span class="k">"explanation":</span> <span class="v">"Invoice unit price $385.00 exceeds PO agreed price $350.00 by 10%. Querying vendor to document discrepancy before final decision."</span></div>
          <div class="step-reveal">
            <span class="reveal-lbl">[ VENDOR ]</span>
            <span>Vendor acknowledges pricing error. Corrected invoice at $350.00 will be reissued.</span>
          </div>
        </div>

        <div class="step-connector">↓</div>

        <div class="step-card">
          <div class="step-top">
            <span class="step-num">Step 2 of 3</span>
            <span class="step-action-badge escalate">ESCALATE</span>
          </div>
          <div class="step-json"><span class="k">"decision":</span> <span class="v">"ESCALATE"</span>, <span class="k">"approved_amount":</span> <span class="n">0.00</span>,
<span class="k">"reason_code":</span> <span class="v">"MANAGER_REVIEW"</span>,
<span class="k">"explanation":</span> <span class="v">"Vendor acknowledged the $385.00 error. Escalating to Finance Manager to confirm rejection and request corrected invoice."</span></div>
          <div class="step-reveal mgr">
            <span class="reveal-lbl mgr">[ MANAGER ]</span>
            <span>Finance Manager confirmed: reject original, request reissued invoice at agreed price $350.00.</span>
          </div>
        </div>

        <div class="step-connector">↓</div>

        <div class="step-card terminal">
          <div class="step-top">
            <span class="step-num">Step 3 — Terminal</span>
            <span class="step-action-badge terminal-ok">REJECT ✓</span>
          </div>
          <div class="step-json"><span class="k">"decision":</span> <span class="v">"REJECT"</span>, <span class="k">"approved_amount":</span> <span class="n">0.00</span>,
<span class="k">"reason_code":</span> <span class="v">"PRICE_DISCREPANCY"</span>,
<span class="k">"explanation":</span> <span class="v">"Invoice price $385.00 vs PO agreed $350.00 — 10% deviation exceeds 1.0% threshold. Vendor confirmed error; corrected invoice required per Policy Rule 4."</span></div>
          <div class="step-score">
            <div class="step-score-num">0.901</div>
            <div class="step-score-lbl">Accumulated reward<br><span style="color:var(--green);font-weight:700">Full investigation + correct terminal decision</span><br><span style="color:var(--dimmer);font-size:11px">vs ~0.40 for skipping straight to REJECT</span></div>
          </div>
        </div>

      </div>
    </div>

    <!-- Reward breakdown -->
    <div class="reward-section fade-in">
      <div class="reward-title">Reward Breakdown — Why 0.901</div>
      <div class="reward-grid">
        <div class="rw-row">
          <div class="rw-label"><span class="rw-name">Decision accuracy</span><span class="rw-val">1.00</span></div>
          <div class="rw-bar-bg"><div class="rw-bar-fill" style="width:100%"></div></div>
        </div>
        <div class="rw-row">
          <div class="rw-label"><span class="rw-name">Amount accuracy</span><span class="rw-val">1.00</span></div>
          <div class="rw-bar-bg"><div class="rw-bar-fill" style="width:100%"></div></div>
        </div>
        <div class="rw-row">
          <div class="rw-label"><span class="rw-name">Reason code</span><span class="rw-val">1.00</span></div>
          <div class="rw-bar-bg"><div class="rw-bar-fill" style="width:100%"></div></div>
        </div>
        <div class="rw-row">
          <div class="rw-label"><span class="rw-name">Explanation quality</span><span class="rw-val">0.90</span></div>
          <div class="rw-bar-bg"><div class="rw-bar-fill" style="width:90%"></div></div>
        </div>
        <div class="rw-row">
          <div class="rw-label"><span class="rw-name">Process bonus <span style="color:var(--teal);font-size:10px">(correct intermediate sequence)</span></span><span class="rw-val" style="color:var(--teal)">+0.10</span></div>
          <div class="rw-bar-bg"><div class="rw-bar-fill" style="width:10%;background:var(--teal)"></div></div>
        </div>
        <div class="rw-row">
          <div class="rw-label"><span class="rw-name">Shortcut REJECT — no investigation</span><span class="rw-val" style="color:var(--red)">~0.40</span></div>
          <div class="rw-bar-bg"><div class="rw-bar-fill" style="width:40%;background:var(--red)"></div></div>
        </div>
      </div>
      <div class="reward-formula">
        <span class="hl">γ<sup>0</sup></span>×0.01 + <span class="hl">γ<sup>1</sup></span>×0.01 + <span class="hl">γ<sup>2</sup></span>×terminal &nbsp;=&nbsp; 0.01 + 0.9×0.01 + 0.81×terminal &nbsp;=&nbsp; <span class="total">0.901</span><br>
        <span style="font-size:10px;color:var(--dimmer)">The reward teaches the right process — not just the right answer. Same terminal decision without the investigation sequence scores ~0.40.</span>
      </div>
    </div>

  </div>
</section>

<!-- ── FOOTER ── -->
<footer>
  <div class="footer-inner">
    <div>
      <div class="footer-brand">
        <svg width="24" height="24" viewBox="0 0 32 32" fill="none"><path d="M16 2L28 8.5V23.5L16 30L4 23.5V8.5L16 2Z" stroke="url(#fg1)" stroke-width="1.5"/><path d="M16 8L22 11.5V18.5L16 22L10 18.5V11.5L16 8Z" fill="url(#fg2)" opacity=".7"/><defs><linearGradient id="fg1" x1="4" y1="2" x2="28" y2="30" gradientUnits="userSpaceOnUse"><stop stop-color="#0ea5e9"/><stop offset="1" stop-color="#14b8a6"/></linearGradient><linearGradient id="fg2" x1="10" y1="8" x2="22" y2="22" gradientUnits="userSpaceOnUse"><stop stop-color="#0ea5e9"/><stop offset="1" stop-color="#14b8a6"/></linearGradient></defs></svg>
        <span>AP Commander</span>
      </div>
      <p class="footer-desc">A multi-agent RL environment for enterprise Accounts Payable workflows. Train LLMs to reason through invoice decisions with rigor — and catch fraud before it costs money.</p>
    </div>
    <div class="footer-links">
      <a href="https://pathikreet-ap-clerk-env.hf.space" target="_blank">Environment Space ↗</a>
      <a href="https://pathikreet-ap-clerk-env.hf.space/docs" target="_blank">API Documentation ↗</a>
      <a href="https://github.com/Vayuputra2401/RL-Agent" target="_blank">GitHub ↗</a>
      <a href="https://huggingface.co/spaces/Pathikreet/ap-commander-training" target="_blank">Training Space ↗</a>
      <a href="https://pathikreet-ap-clerk-env.hf.space/tasks" target="_blank">Task Library ↗</a>
    </div>
  </div>
  <div class="footer-bottom">
    <span>Pathikreet Chowdhury · Anubhav Bhattacharya · Radhika Ravi</span>
    <span>Meta PyTorch OpenEnv × Scaler School of Technology · 2026</span>
  </div>
</footer>

<script>
const $=id=>document.getElementById(id);

// ── Scroll effects ──
window.addEventListener('scroll',()=>{$('nav').classList.toggle('scrolled',window.scrollY>50);});
const io=new IntersectionObserver(entries=>{entries.forEach(e=>{if(e.isIntersecting){e.target.classList.add('visible');io.unobserve(e.target);}});},{threshold:.15});
document.querySelectorAll('.fade-in').forEach(el=>io.observe(el));

// ── Stats count-up ──
let statsAnimated=false;
const statsIO=new IntersectionObserver(entries=>{if(entries[0].isIntersecting&&!statsAnimated){statsAnimated=true;loadStats();}},{threshold:.3});
statsIO.observe($('stats'));
function countUp(el,target,duration=1200){
  const isFloat=String(target).includes('.');let start=0,t0=null;
  function frame(t){if(!t0)t0=t;const p=Math.min((t-t0)/duration,1);const ease=1-Math.pow(1-p,3);const v=start+(target-start)*ease;el.textContent=isFloat?v.toFixed(3):Math.round(v).toLocaleString();if(p<1)requestAnimationFrame(frame);}
  requestAnimationFrame(frame);
}

// ── Boot ──
async function boot(){checkHealth();setInterval(loadStats,30000);}

async function checkHealth(){
  try{await fetch('/health');$('health-dot').className='health-dot ok';$('health-text').textContent='Live';}
  catch{$('health-dot').className='health-dot err';$('health-text').textContent='Offline';}
}

async function loadStats(){
  try{
    const d=await(await fetch('/stats')).json();
    const ep=d.completed_episodes||d.total_episodes||0;
    const ms=d.mean_score||0;
    countUp($('stat-episodes'),ep);
    if(ms>0){$('stat-score').style.cssText='font-size:40px;font-weight:900;letter-spacing:-2px;background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1';countUp($('stat-score'),ms,1400);}
  }catch{}
}

boot();
</script>
</body>
</html>"""
