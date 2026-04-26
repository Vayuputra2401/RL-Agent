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

/* ── DEMO SECTION ── */
#demo{background:var(--bg)}
.demo-controls{display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap;margin-bottom:28px;background:var(--glass);backdrop-filter:blur(12px);border:1px solid var(--border);border-radius:12px;padding:20px 24px}
.field{display:flex;flex-direction:column;gap:5px}
.field label{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.7px;color:var(--dim)}
select,input[type=number],input[type=text],textarea{background:rgba(2,8,23,0.8);border:1px solid var(--border);color:var(--text);border-radius:8px;padding:9px 12px;font-size:13px;outline:none;font-family:inherit;transition:border-color .2s}
select:focus,input:focus,textarea:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(14,165,233,0.1)}
select{min-width:240px;cursor:pointer}
input[type=number]{width:90px}
textarea{resize:vertical;min-height:72px;width:100%}
.btn-run{background:var(--grad);color:#020817;font-weight:800;font-size:14px;padding:10px 22px;border-radius:8px;border:none;cursor:pointer;transition:all .2s;display:flex;align-items:center;gap:8px;white-space:nowrap}
.btn-run:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 8px 24px rgba(14,165,233,0.3)}
.btn-run:disabled{opacity:.45;cursor:not-allowed;transform:none}
.btn-submit{background:rgba(34,197,94,0.15);color:var(--green);border:1px solid rgba(34,197,94,0.35);font-weight:700;font-size:14px;padding:10px 22px;border-radius:8px;cursor:pointer;transition:all .2s;display:flex;align-items:center;gap:8px}
.btn-submit:hover:not(:disabled){background:rgba(34,197,94,0.25);box-shadow:0 0 20px rgba(34,197,94,0.2)}
.btn-submit:disabled{opacity:.4;cursor:not-allowed}

.demo-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.demo-grid.single{grid-template-columns:1fr}

/* Doc cards */
.doc-card{background:var(--glass);backdrop-filter:blur(12px);border:1px solid var(--border);border-radius:12px;padding:20px;transition:border-color .2s}
.doc-card:hover{border-color:rgba(14,165,233,0.3)}
.doc-label{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--accent);margin-bottom:12px;display:flex;align-items:center;gap:6px}
.doc-label::after{content:'';flex:1;height:1px;background:var(--border)}
.inv-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;gap:12px;flex-wrap:wrap}
.inv-id{font-size:11px;color:var(--dim);font-family:monospace}
.inv-vendor{font-size:16px;font-weight:800;color:var(--text);letter-spacing:-.3px}
.inv-total{font-size:28px;font-weight:900;background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-1px;white-space:nowrap}
table.items{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px}
table.items th{color:var(--dim);font-weight:600;text-align:left;padding:3px 8px 5px 0;border-bottom:1px solid var(--border);font-size:11px;text-transform:uppercase;letter-spacing:.5px}
table.items td{padding:4px 8px 4px 0;border-bottom:1px solid rgba(14,165,233,0.06)}
table.items tr:last-child td{border-bottom:none}
.freight-ok{color:var(--dim)}
.freight-warn{color:var(--yellow);font-weight:600}
.freight-over{color:var(--red);font-weight:700}
.badge{display:inline-flex;align-items:center;padding:2px 8px;border-radius:5px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.4px}
.badge-open{background:rgba(34,197,94,0.12);color:var(--green);border:1px solid rgba(34,197,94,0.3)}
.badge-closed{background:rgba(71,85,105,0.2);color:var(--dim);border:1px solid var(--border)}
.po-row{margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid var(--border)}
.po-row:last-child{margin-bottom:0;padding-bottom:0;border-bottom:none}
.po-header{display:flex;align-items:center;gap:8px;margin-bottom:6px;flex-wrap:wrap}
.po-num{font-size:13px;font-weight:700;font-family:monospace;color:var(--text)}
.po-vendor{font-size:12px;color:var(--dim)}
.grn-row{margin-bottom:8px}
.grn-row:last-child{margin-bottom:0}
.grn-id{font-size:12px;font-weight:700;font-family:monospace;color:var(--teal);margin-bottom:4px}

.alert-duplicate{display:flex;align-items:center;gap:10px;background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);border-radius:8px;padding:10px 14px;font-size:13px;color:var(--red);font-weight:600;margin-bottom:16px}
.context-note{display:flex;gap:10px;background:rgba(14,165,233,0.07);border-left:3px solid var(--accent);border-radius:0 8px 8px 0;padding:10px 14px;font-size:13px;color:var(--text);margin-bottom:8px}
.step-indicator{display:inline-flex;align-items:center;gap:8px;background:rgba(139,92,246,0.12);border:1px solid rgba(139,92,246,0.3);border-radius:20px;padding:5px 14px;font-size:12px;color:var(--purple);font-weight:700;margin-bottom:16px}
.step-dots{display:flex;gap:4px;margin-left:4px}
.step-dot{width:7px;height:7px;border-radius:50%;background:rgba(139,92,246,0.3)}
.step-dot.done{background:var(--purple)}
.step-dot.active{background:var(--accent);box-shadow:0 0 6px var(--accent)}

/* Action form */
.action-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.span2{grid-column:1/-1}
.form-actions{display:flex;align-items:center;gap:12px;margin-top:16px}

/* Result */
.result-score-row{display:flex;align-items:center;gap:20px;margin-bottom:20px}
.score-number{font-size:52px;font-weight:900;letter-spacing:-2px;line-height:1}
.score-bar-wrap{flex:1}
.score-bar-bg{height:8px;background:rgba(255,255,255,0.06);border-radius:4px;overflow:hidden;margin-bottom:6px}
.score-bar-fill{height:100%;border-radius:4px;transition:width .5s cubic-bezier(.4,0,.2,1)}
.score-verdict{font-size:13px;color:var(--dim)}
.comp-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px;margin-bottom:16px}
.comp-item{background:rgba(14,165,233,0.05);border:1px solid var(--border);border-radius:8px;padding:10px;text-align:center}
.comp-name{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;font-weight:600}
.comp-val{font-size:16px;font-weight:800;font-family:monospace}
.done-banner{background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);border-radius:8px;padding:12px 16px;color:var(--green);font-weight:700;font-size:14px;text-align:center;margin-top:8px}
.continue-banner{background:rgba(14,165,233,0.07);border:1px solid var(--border);border-radius:8px;padding:12px 16px;color:var(--accent);font-size:13px;text-align:center;margin-top:8px}

/* Policy accordion */
.policy-toggle{width:100%;background:none;border:1px solid var(--border);color:var(--dim);padding:10px 14px;border-radius:8px;cursor:pointer;text-align:left;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.7px;display:flex;justify-content:space-between;align-items:center;transition:border-color .2s,color .2s;margin-top:12px}
.policy-toggle:hover{border-color:var(--accent);color:var(--accent)}
.policy-box{background:rgba(2,8,23,0.5);border:1px solid var(--border);border-radius:8px;padding:12px;font-size:11px;color:var(--dim);white-space:pre-wrap;max-height:140px;overflow-y:auto;margin-top:6px;font-family:monospace;line-height:1.6;display:none}

/* Task chips */
.task-chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:20px}
.task-chip{display:inline-flex;align-items:center;gap:6px;padding:6px 12px;border-radius:20px;border:1px solid var(--border);background:var(--glass);font-size:11px;cursor:pointer;transition:all .2s;backdrop-filter:blur(8px)}
.task-chip:hover,.task-chip.active{border-color:var(--accent);background:rgba(14,165,233,0.1);color:var(--text)}
.task-chip .diff-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}

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
    <a href="#demo">Live Demo</a>
    <a href="/docs" target="_blank">API Docs ↗</a>
    <a href="https://github.com/Vayuputra2401/RL-Agent" target="_blank">GitHub ↗</a>
    <a href="#demo" class="nav-cta">Try Demo</a>
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
    <a href="#demo" class="btn btn-primary">▶ &nbsp;Try Live Demo</a>
    <a href="/docs" target="_blank" class="btn btn-secondary">API Documentation ↗</a>
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
      <div class="section-label">Live Demo</div>
      <h2 class="section-title">Try an <span class="grad-text">Episode Now</span></h2>
      <p class="section-sub">Pick any task, generate a fresh invoice scenario, submit your decision, and see the reward breakdown — no code needed.</p>
    </div>

    <!-- Controls -->
    <div class="demo-controls">
      <div class="field">
        <label>Task</label>
        <select id="task-select" onchange="onTaskChange()"><option>Loading…</option></select>
      </div>
      <div class="field">
        <label>Seed</label>
        <input id="seed-input" type="number" value="42" min="1" max="9999">
      </div>
      <button id="new-ep-btn" class="btn-run" onclick="newEpisode()">
        <span id="btn-icon">▶</span> New Episode
      </button>
    </div>

    <p id="task-desc" class="hidden" style="font-size:13px;color:var(--dim);margin-bottom:16px;padding:10px 16px;background:rgba(14,165,233,0.06);border-radius:8px;border-left:3px solid var(--accent);"></p>

    <!-- Task chips -->
    <div id="task-chips" class="task-chips"></div>

    <!-- Episode content -->
    <div id="episode-area" class="hidden" style="margin-top:28px;">

      <div id="step-indicator" class="step-indicator hidden">
        <span>Step</span>
        <strong id="step-num">1</strong>
        <span>of</span>
        <strong id="step-max">3</strong>
        <div id="step-dots" class="step-dots"></div>
      </div>

      <div id="dupe-alert" class="alert-duplicate hidden">
        ⚠ DUPLICATE DETECTED — This invoice ID already appears in the paid ledger
      </div>

      <div id="context-notes-wrap"></div>

      <div class="demo-grid" id="obs-grid">

        <!-- Left col: invoice + POs + GRNs + policy -->
        <div style="display:flex;flex-direction:column;gap:14px;">
          <div class="doc-card">
            <div class="doc-label">Invoice</div>
            <div id="invoice-body"></div>
          </div>
          <div class="doc-card">
            <div class="doc-label">Purchase Orders</div>
            <div id="po-body"></div>
          </div>
          <div class="doc-card">
            <div class="doc-label">Goods Receipts</div>
            <div id="grn-body"></div>
          </div>
          <button class="policy-toggle" onclick="togglePolicy()">Company Policy <span id="pol-arrow">▼</span></button>
          <div id="policy-box" class="policy-box"></div>
        </div>

        <!-- Right col: action form + result -->
        <div style="display:flex;flex-direction:column;gap:14px;">

          <!-- Action form -->
          <div class="doc-card" id="action-card">
            <div class="doc-label">Your Decision</div>
            <div class="action-grid">
              <div class="field">
                <label>Decision</label>
                <select id="a-decision">
                  <option>APPROVE_FULL</option>
                  <option>APPROVE_PARTIAL</option>
                  <option selected>REJECT</option>
                  <option>QUERY_VENDOR</option>
                  <option>ESCALATE</option>
                  <option>HOLD</option>
                </select>
              </div>
              <div class="field">
                <label>Amount ($)</label>
                <input id="a-amount" type="number" value="0.00" min="0" step="0.01">
              </div>
              <div class="field span2">
                <label>Reason Code</label>
                <select id="a-reason">
                  <option>MATCH_CONFIRMED</option><option>QUANTITY_MISMATCH</option>
                  <option>PRICE_DISCREPANCY</option><option>POLICY_VIOLATION</option>
                  <option>NO_PO_FOUND</option><option selected>DUPLICATE_INVOICE</option>
                  <option>VENDOR_MISMATCH</option><option>TAX_DISCREPANCY</option>
                  <option>PENDING_CLARIFICATION</option><option>MANAGER_REVIEW</option>
                </select>
              </div>
              <div class="field span2">
                <label>Explanation &nbsp;<span style="color:var(--dimmer);text-transform:none;letter-spacing:0;font-weight:400;">(cite specific $ amounts)</span></label>
                <textarea id="a-explain" placeholder="e.g. Invoice $4,200 matches PO-991 line items but INV-2847 already appears in the paid ledger — rejecting as duplicate payment."></textarea>
              </div>
            </div>
            <div class="form-actions">
              <button id="submit-btn" class="btn-submit" onclick="submitAction()">Submit Action</button>
              <span id="submit-spin" class="spinner hidden"></span>
            </div>
          </div>

          <!-- Result -->
          <div class="doc-card hidden" id="result-card">
            <div class="doc-label">Result</div>
            <div class="result-score-row">
              <div id="score-num" class="score-number">—</div>
              <div class="score-bar-wrap">
                <div class="score-bar-bg">
                  <div id="score-bar" class="score-bar-fill" style="width:0%"></div>
                </div>
                <div id="score-verdict" class="score-verdict"></div>
              </div>
            </div>
            <div id="comp-grid" class="comp-grid"></div>
            <div id="result-banner"></div>
          </div>

        </div>
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
      <a href="/docs" target="_blank">API Documentation ↗</a>
      <a href="https://github.com/Vayuputra2401/RL-Agent" target="_blank">GitHub ↗</a>
      <a href="https://huggingface.co/spaces/Pathikreet/ap-commander-training" target="_blank">Training Space ↗</a>
      <a href="/tasks" target="_blank">Task Library ↗</a>
      <a href="/health" target="_blank">Health Check ↗</a>
    </div>
  </div>
  <div class="footer-bottom">
    <span>Pathikreet Chowdhury · Anubhav Bhattacharya · Radhika Ravi</span>
    <span>Meta PyTorch OpenEnv × Scaler School of Technology · 2026</span>
  </div>
</footer>

<script>
const $=id=>document.getElementById(id);
const show=id=>$(id).classList.remove('hidden');
const hide=id=>$(id).classList.add('hidden');
const fmt$=n=>'$'+parseFloat(n||0).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
let SESSION_ID=null, TASKS=[], FREIGHT_CAP=50;

// ── Scroll effects ────────────────────────────────────────────────────────────
window.addEventListener('scroll',()=>{
  $('nav').classList.toggle('scrolled',window.scrollY>50);
});

const io=new IntersectionObserver(entries=>{
  entries.forEach(e=>{if(e.isIntersecting){e.target.classList.add('visible');io.unobserve(e.target);}});
},{threshold:.15});
document.querySelectorAll('.fade-in').forEach(el=>io.observe(el));

// ── Stats count-up ────────────────────────────────────────────────────────────
let statsAnimated=false;
const statsIO=new IntersectionObserver(entries=>{
  if(entries[0].isIntersecting&&!statsAnimated){statsAnimated=true;loadStats();}
},{threshold:.3});
statsIO.observe($('stats'));

function countUp(el,target,duration=1200){
  const isFloat=String(target).includes('.');
  let start=0,t0=null;
  function frame(t){
    if(!t0)t0=t;
    const p=Math.min((t-t0)/duration,1);
    const ease=1-Math.pow(1-p,3);
    const v=start+(target-start)*ease;
    el.textContent=isFloat?v.toFixed(3):Math.round(v).toLocaleString();
    if(p<1)requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// ── Boot ─────────────────────────────────────────────────────────────────────
async function boot(){
  checkHealth();
  loadTasks();
  setInterval(loadStats,30000);
}

async function checkHealth(){
  try{
    const r=await fetch('/health');
    const d=await r.json();
    $('health-dot').className='health-dot ok';
    $('health-text').textContent='Live';
  }catch{
    $('health-dot').className='health-dot err';
    $('health-text').textContent='Offline';
  }
}

async function loadStats(){
  try{
    const r=await fetch('/stats');
    const d=await r.json();
    const ep=d.completed_episodes||d.total_episodes||0;
    const ms=d.mean_score||0;
    countUp($('stat-episodes'),ep);
    if(ms>0){
      $('stat-score').style.cssText='font-size:40px;font-weight:900;letter-spacing:-2px;background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1';
      countUp($('stat-score'),ms,1400);
    }
  }catch{}
}

async function loadTasks(){
  try{
    const r=await fetch('/tasks');
    TASKS=await r.json();
    buildDropdown();
    buildChips();
    onTaskChange();
  }catch(e){console.error(e);}
}

function buildDropdown(){
  const sel=$('task-select'); sel.innerHTML='';
  const order=['easy','medium','hard','long-horizon','oversight'];
  const groups={};
  TASKS.forEach(t=>{const d=t.difficulty||'easy';(groups[d]=groups[d]||[]).push(t);});
  const labels={easy:'Easy',medium:'Medium',hard:'Hard','long-horizon':'Long-Horizon',oversight:'Oversight'};
  order.forEach(d=>{
    if(!groups[d]?.length)return;
    const g=document.createElement('optgroup'); g.label=labels[d]||d;
    groups[d].forEach(t=>{const o=document.createElement('option');o.value=t.id;o.textContent=t.name||t.id;g.appendChild(o);});
    sel.appendChild(g);
  });
}

function buildChips(){
  const wrap=$('task-chips'); wrap.innerHTML='';
  const colors={easy:'#22c55e',medium:'#eab308',hard:'#ef4444','long-horizon':'#8b5cf6',oversight:'#0ea5e9'};
  TASKS.slice(0,12).forEach(t=>{
    const chip=document.createElement('div');
    chip.className='task-chip'; chip.dataset.id=t.id;
    const d=(t.difficulty||'easy').replace('-horizon','');
    chip.innerHTML=`<span class="diff-dot" style="background:${colors[t.difficulty]||'#94a3b8'};"></span>${t.name||t.id}`;
    chip.onclick=()=>{$('task-select').value=t.id;onTaskChange();newEpisode();};
    wrap.appendChild(chip);
  });
  if(TASKS.length>12){
    const more=document.createElement('div');
    more.className='task-chip';more.style.opacity='.5';
    more.textContent=`+${TASKS.length-12} more`;
    wrap.appendChild(more);
  }
}

function onTaskChange(){
  const t=TASKS.find(x=>x.id===$('task-select').value);
  if(!t)return;
  document.querySelectorAll('.task-chip').forEach(c=>c.classList.toggle('active',c.dataset.id===t.id));
  const desc=$('task-desc');
  const txt=[t.notes,t.expected_decision?'Expected: '+t.expected_decision:''].filter(Boolean).join('  ·  ');
  if(txt){desc.textContent=txt;show('task-desc');}else hide('task-desc');
}

// ── New Episode ───────────────────────────────────────────────────────────────
async function newEpisode(){
  const task_id=$('task-select').value;
  const seed=parseInt($('seed-input').value)||42;
  const btn=$('new-ep-btn');
  btn.disabled=true; $('btn-icon').textContent='⏳';
  hide('episode-area'); hide('result-card');

  try{
    const r=await fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id,seed})});
    if(!r.ok)throw new Error(await r.text());
    const data=await r.json();
    SESSION_ID=data.session_id;
    renderObs(data.observation);
    show('episode-area');
    $('episode-area').scrollIntoView({behavior:'smooth',block:'nearest'});
  }catch(e){alert('Error: '+e.message);}
  finally{btn.disabled=false; $('btn-icon').textContent='▶';}
}

function renderObs(obs){
  FREIGHT_CAP=obs.freight_cap||50;
  const inv=obs.invoice||{};
  const step=obs.step_count||0, max=obs.max_steps||1;

  // Step indicator
  if(max>1){
    $('step-num').textContent=step+1; $('step-max').textContent=max;
    const dots=$('step-dots'); dots.innerHTML='';
    for(let i=0;i<Math.min(max,12);i++){const d=document.createElement('span');d.className='step-dot'+(i<step?' done':i===step?' active':'');dots.appendChild(d);}
    show('step-indicator');
  }else hide('step-indicator');

  // Duplicate alert
  const paid=obs.paid_invoice_ids||[];
  if(paid.includes(inv.invoice_id))show('dupe-alert'); else hide('dupe-alert');

  // Context notes
  const notes=obs.context_notes||[];
  $('context-notes-wrap').innerHTML=notes.map(n=>`<div class="context-note">📋 &nbsp;${n}</div>`).join('');

  // Invoice
  const lines=(inv.line_items||[]).map(li=>`<tr><td>${li.description||'Item'}</td><td style="text-align:right">${li.quantity||0}</td><td style="text-align:right">${fmt$(li.unit_price)}</td><td style="text-align:right;font-weight:600">${fmt$((li.quantity||0)*(li.unit_price||0))}</td></tr>`).join('');
  const fr=parseFloat(inv.freight_charge||0);
  const frCls=fr>FREIGHT_CAP?'freight-over':fr>FREIGHT_CAP*.9?'freight-warn':'freight-ok';
  const frNote=fr>FREIGHT_CAP?` <span style="font-size:10px">⚠ over ${fmt$(FREIGHT_CAP)} cap</span>`:'';
  $('invoice-body').innerHTML=`
    <div class="inv-header"><div><div class="inv-id">${inv.invoice_id||'—'}</div><div class="inv-vendor">${inv.vendor_name||'Unknown'}</div></div><div class="inv-total">${fmt$(inv.invoice_total)}</div></div>
    <table class="items"><thead><tr><th>Item</th><th style="text-align:right">Qty</th><th style="text-align:right">Unit</th><th style="text-align:right">Total</th></tr></thead>
    <tbody>${lines}<tr class="${frCls}"><td colspan="3">Freight${frNote}</td><td style="text-align:right">${fmt$(fr)}</td></tr></tbody></table>`;

  // POs
  const pos=obs.purchase_orders||[];
  $('po-body').innerHTML=pos.length?pos.map(p=>{
    const stat=(p.status||'UNKNOWN').toUpperCase();
    const badge=`<span class="badge badge-${stat==='OPEN'?'open':'closed'}">${stat}</span>`;
    const plines=(p.lines||[]).map(l=>`<div style="font-size:11px;color:var(--dim);margin-top:3px;font-family:monospace">${l.description||'Item'}&nbsp;·&nbsp;qty ${l.ordered_quantity}&nbsp;@&nbsp;${fmt$(l.agreed_unit_price)}</div>`).join('');
    return `<div class="po-row"><div class="po-header"><span class="po-num">${p.po_number||'PO'}</span>${badge}<span class="po-vendor">${p.vendor_name||''}</span></div>${plines}</div>`;
  }).join(''):'<span style="color:var(--dim);font-size:12px">No purchase orders</span>';

  // GRNs
  const grns=obs.goods_receipts||[];
  $('grn-body').innerHTML=grns.length?grns.map(g=>{
    const gl=(g.lines||[]).map(l=>`<div style="font-size:11px;color:var(--dim);margin-top:3px;font-family:monospace">${l.description||'Item'}&nbsp;·&nbsp;received&nbsp;<strong style="color:var(--teal)">${l.received_quantity}</strong></div>`).join('');
    return `<div class="grn-row"><div class="grn-id">${g.grn_id||'GRN'}</div>${gl}</div>`;
  }).join(''):'<span style="color:var(--dim);font-size:12px">No goods receipts</span>';

  // Policy
  $('policy-box').textContent=obs.company_policy||'';
  $('policy-box').style.display='none'; $('pol-arrow').textContent='▼';

  // Reset action form
  const total=parseFloat(inv.invoice_total||0);
  $('a-amount').value=total.toFixed(2);
  $('a-explain').value='';
  hide('result-card');
  show('action-card');
}

function togglePolicy(){
  const box=$('policy-box');
  const showing=box.style.display==='block';
  box.style.display=showing?'none':'block';
  $('pol-arrow').textContent=showing?'▼':'▲';
}

// ── Submit ────────────────────────────────────────────────────────────────────
async function submitAction(){
  if(!SESSION_ID)return;
  const action={decision:$('a-decision').value,approved_amount:parseFloat($('a-amount').value)||0,reason_code:$('a-reason').value,explanation:$('a-explain').value.trim()||'No explanation.'};
  $('submit-btn').disabled=true; show('submit-spin');
  try{
    const r=await fetch('/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:SESSION_ID,action})});
    if(!r.ok)throw new Error(await r.text());
    const data=await r.json();
    renderResult(data);
    if(!data.done&&data.observation)renderObs(data.observation);
  }catch(e){alert('Error: '+e.message);}
  finally{$('submit-btn').disabled=false; hide('submit-spin');}
}

function renderResult(data){
  const reward=data.reward||{};
  const score=parseFloat(reward.score||0);
  const col=score>=0.75?'var(--green)':score>=0.45?'var(--yellow)':'var(--red)';

  $('score-num').textContent=score.toFixed(3);
  $('score-num').style.color=col;
  $('score-bar').style.width=(score*100).toFixed(1)+'%';
  $('score-bar').style.background=col;
  $('score-verdict').textContent=score>=0.75?'Strong performance — reward signal satisfied':score>=0.45?'Partial credit — check amount and explanation':'Low score — incorrect decision or missing citations';

  const comps=reward.components||{};
  $('comp-grid').innerHTML=Object.entries(comps).map(([k,v])=>{
    const val=typeof v==='number'?v.toFixed(3):v;
    const c=typeof v==='number'?(v>0.01?'var(--green)':v<0?'var(--red)':'var(--dim)'):'var(--text)';
    return `<div class="comp-item"><div class="comp-name">${k.replace(/_/g,' ')}</div><div class="comp-val" style="color:${c}">${val}</div></div>`;
  }).join('');

  if(data.done){
    $('result-banner').innerHTML=`<div class="done-banner">✓ Episode complete &nbsp;·&nbsp; Final score: ${score.toFixed(3)}</div>`;
    hide('action-card');
  }else{
    const s=data.observation?.step_count??'?',m=data.observation?.max_steps??'?';
    $('result-banner').innerHTML=`<div class="continue-banner">Intermediate action recorded &nbsp;·&nbsp; Continue with step ${(s||0)+1} of ${m}</div>`;
  }
  show('result-card');
  $('result-card').scrollIntoView({behavior:'smooth',block:'nearest'});
  loadStats();
}

boot();
</script>
</body>
</html>"""
