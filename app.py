import base64, pathlib
import gradio as gr
from rag import answer

NAVY, SURFACE, RAISED, LINE = "#1b2231", "#222a3a", "#2a3345", "#383e4b"
RED, RED_DEEP, INK, MUTED = "#b40013", "#971814", "#ffffff", "#a7aebb"

logo = pathlib.Path("logo.png")
logo_src = f"data:image/png;base64,{base64.b64encode(logo.read_bytes()).decode()}" if logo.exists() else ""

theme = gr.themes.Base(
    font=[gr.themes.GoogleFont("IBM Plex Sans Arabic"), "system-ui", "sans-serif"],
    radius_size="md",
).set(
    body_background_fill=NAVY, body_background_fill_dark=NAVY,
    body_text_color=INK, body_text_color_dark=INK,
    body_text_color_subdued=MUTED, body_text_color_subdued_dark=MUTED,
    background_fill_primary=SURFACE, background_fill_primary_dark=SURFACE,
    background_fill_secondary=RAISED, background_fill_secondary_dark=RAISED,
    border_color_primary=LINE, border_color_primary_dark=LINE,
    block_background_fill=SURFACE, block_background_fill_dark=SURFACE, block_border_color=LINE,
    input_background_fill=RAISED, input_background_fill_dark=RAISED, input_border_color=LINE,
    button_primary_background_fill=RED, button_primary_background_fill_hover=RED_DEEP,
    button_primary_text_color="#ffffff",
    color_accent=RED, color_accent_soft="rgba(180,0,19,.18)", link_text_color="#ff6b7a",
)

CSS = """
:root { --mm-navy:#1b2231; --mm-surface:#222a3a; --mm-raised:#2a3345; --mm-line:#383e4b;
        --mm-red:#b40013; --mm-red-deep:#971814; --mm-ink:#ffffff; --mm-muted:#a7aebb; }
footer { display:none !important; }
.gradio-container { max-width:960px !important; margin:0 auto !important; padding-inline:16px !important; }

#mm-header { display:flex; align-items:center; justify-content:space-between; gap:16px;
             padding:18px 0 14px; border-bottom:1px solid var(--mm-line); }
.mm-brand { display:flex; align-items:center; gap:14px; }
.mm-brand img { width:52px; height:52px; border-radius:10px; display:block; }
.mm-brand h1 { margin:0; font-size:1.6rem; font-weight:700; line-height:1.2; color:var(--mm-ink); }
.mm-brand p { margin:2px 0 0; font-size:.85rem; color:var(--mm-muted); }
.mm-pill { display:inline-flex; align-items:center; gap:8px; background:var(--mm-red); color:#fff;
           font-weight:700; font-size:.9rem; padding:8px 14px; border-radius:6px; white-space:nowrap; }
.mm-pill::before { content:""; width:8px; height:8px; border-radius:50%; background:#fff; }

#mm-strip { display:flex; align-items:stretch; margin:14px 0 6px; overflow-x:auto;
            white-space:nowrap; scrollbar-width:none; }
.mm-tab { background:var(--mm-red); color:#fff; font-weight:700; padding:8px 18px; border-radius:4px; }
.mm-doc { display:inline-flex; align-items:center; gap:8px; padding:8px 14px; font-weight:600;
          font-size:.95rem; color:var(--mm-ink); }
.mm-doc::after { content:""; width:7px; height:7px; border-radius:50%; background:var(--mm-red); }

#mm-chat { border:1px solid var(--mm-line) !important; border-radius:12px !important;
           background:var(--mm-surface) !important; }
#mm-chat .bot-row { background:var(--mm-raised) !important; border:1px solid var(--mm-line) !important; }
#mm-chat .user-row { background:transparent !important; border:0 !important; }
#mm-chat .user { background-color:var(--mm-red) !important; border-color:var(--mm-red) !important;
                 box-shadow:none !important; }
#mm-chat .user-row .prose, #mm-chat .user-row .prose * { color:#fff !important; }
#mm-chat .prose { color:var(--mm-ink) !important; line-height:1.75; text-align:start; }
#mm-chat .prose blockquote { margin:0 0 12px; padding:10px 14px; border:0;
    border-inline-start:4px solid var(--mm-red); background:rgba(180,0,19,.14);
    border-radius:6px; color:#fff; }
#mm-chat .prose ul { color:var(--mm-muted); font-size:.88em; margin-top:4px; }
#mm-chat .example { background:var(--mm-raised) !important; border:1px solid var(--mm-line) !important;
    color:var(--mm-ink) !important; border-radius:8px !important; transition:border-color .15s; }
#mm-chat .example:hover { border-color:var(--mm-red) !important; }

.html-container { padding-inline:0 !important; }
#mm-note, #mm-note * { color:var(--mm-muted) !important; }
#mm-note { text-align:center; color:var(--mm-muted); font-size:.8rem; line-height:1.6; padding:6px 0 18px; }
:focus-visible { outline:2px solid #ff4d5e !important; outline-offset:2px; }

@media (prefers-reduced-motion: no-preference) {
  #mm-header { animation:mm-in .5s ease-out both; }
  .mm-tab { animation:mm-wipe .6s .15s ease-out both; }
  .mm-doc { animation:mm-in .4s ease-out both; }
  .mm-doc:nth-child(2) { animation-delay:.35s; }
  .mm-doc:nth-child(3) { animation-delay:.45s; }
  .mm-doc:nth-child(4) { animation-delay:.55s; }
}
@keyframes mm-in { from { opacity:0; transform:translateY(-6px); } to { opacity:1; transform:none; } }
@keyframes mm-wipe { from { clip-path:inset(0 0 0 100%); } to { clip-path:inset(0); } }

@media (max-width:640px) {
  .mm-brand h1 { font-size:1.25rem; }
  .mm-brand img { width:42px; height:42px; }
  .mm-pill { font-size:.8rem; padding:6px 10px; }
}
"""

# each message / the input picks RTL or LTR from its own first strong character
JS = """() => {
  const fix = () => document.querySelectorAll('#mm-chat .prose, textarea')
    .forEach(e => { if (!e.dir) e.dir = 'auto'; });
  new MutationObserver(fix).observe(document.body, { childList: true, subtree: true });
  fix();
}"""

HEADER = f"""
<header id="mm-header" dir="rtl">
  <div class="mm-brand">
    <img src="{logo_src}" alt="شعار قناة المملكة">
    <div>
      <h1>مساعد المملكة</h1>
      <p lang="en" dir="ltr">Digital Expansion Initiative assistant</p>
    </div>
  </div>
  <span class="mm-pill">يجيب من الوثائق فقط</span>
</header>
<nav id="mm-strip" dir="rtl" aria-label="المصادر">
  <span class="mm-tab">المصادر</span>
  <span class="mm-doc" title="Project Overview">نظرة عامة على المشروع</span>
  <span class="mm-doc" title="Budget &amp; Timeline">الميزانية والجدول الزمني</span>
  <span class="mm-doc" title="Team &amp; Governance">الفريق والحوكمة</span>
</nav>
"""

PLACEHOLDER = """### أهلاً بك
اسأل عن الميزانية أو الجدول الزمني أو فريق المشروع.

Ask about the budget, timeline or project team, in Arabic or English."""

with gr.Blocks(title="مساعد المملكة | Almamlaka TV Assistant", fill_height=True) as demo:
    gr.HTML(HEADER)
    gr.ChatInterface(
        fn=answer,
        chatbot=gr.Chatbot(elem_id="mm-chat", height=540, show_label=False,
                           placeholder=PLACEHOLDER, latex_delimiters=[], sanitize_html=True),
        textbox=gr.Textbox(elem_id="mm-ask", placeholder="اكتب سؤالك هنا… Type your question",
                           max_length=1000, show_label=False),
        examples=["ما هي الميزانية الإجمالية للمشروع؟", "When is the public launch?",
                  "Who do I contact for a technical escalation?", "متى موعد الإطلاق؟"],
        submit_btn="إرسال",
        cache_examples=False,
        concurrency_limit=4,
        fill_height=True,
    )
    gr.HTML('<div id="mm-note" dir="rtl">كل إجابة مرفقة باسم الوثيقة والصفحة والقسم.<br>'
            '<span dir="ltr">Every answer cites its document, page and section.</span></div>')

if __name__ == "__main__":
    demo.launch(theme=theme, css=CSS, js=JS, favicon_path="logo.png" if logo.exists() else None)
