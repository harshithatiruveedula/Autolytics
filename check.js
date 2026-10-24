
'use strict';
let args={}, state={page:1,query:'',file:null,busyQuery:false,busyUpload:false,busyEda:false};
const root=document.getElementById('root');
let eventCounter=0;
function send(value){value.event_id=String(Date.now())+'-'+(++eventCounter)+'-'+Math.random().toString(36).slice(2,8);window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:setComponentValue',value,dataType:'json'},'*');}
function ready(){window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:componentReady',apiVersion:1},'*');}
function height(){const h=Math.max(document.body.scrollHeight,document.documentElement.scrollHeight,60);window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:setFrameHeight',height:h},'*');}
function esc(x){return String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function fmt(x){if(x===null||x===undefined||x==='')return '<span class="blank">Missing</span>';return esc(x);}
function typeName(t){if(/int|float|double|decimal/i.test(t||''))return 'Number';if(/date|time/i.test(t||''))return 'Date';return 'Text';}

/* ── Nav item helper ─────────────────────────────────────────────────── */
// "upload" and "query" are sections on the home page, not separate pages.
// They highlight as active when the current nav matches them, but the home
// page content is always fully rendered when nav is home/upload/query.
function isHomePage(nav){return nav==='home'||nav==='upload'||nav==='query';}
function navItem(label,key,icon){
  const active=args.nav===key||(key==='home'&&isHomePage(args.nav)&&args.nav!=='upload'&&args.nav!=='query')
               ?args.nav===key:args.nav===key;
  const busy=state.busyQuery&&key==='query';
  return `<div class="nav ${args.nav===key?'active':''} ${busy?'busy':''}" data-nav="${key}" title="${label}"><span class="navicon" aria-hidden="true">${icon}</span><span class="navlabel">${label}</span></div>`;
}

/* ── Upload status ───────────────────────────────────────────────────── */
function uploadStatus(title,detail){
  const drop=document.getElementById('drop');if(!drop)return;
  drop.classList.add('uploading');
  const strong=drop.querySelector('strong'),smalls=drop.querySelectorAll('small'),btn=document.getElementById('chooseFile');
  if(strong)strong.textContent=title;if(smalls[0])smalls[0].textContent=detail||'Please wait…';
  if(btn){btn.disabled=true;btn.textContent='Processing…';btn.style.opacity='.7';}
}

/* ── Query submission ────────────────────────────────────────────────── */
function sendQuery(value){
  if(state.busyQuery)return;
  state.busyQuery=true;state.query=value;
  const run=document.getElementById('run');
  if(run){run.disabled=true;run.textContent='Running…';run.style.opacity='.75';}
  document.querySelectorAll('[data-nav="query"]').forEach(el=>el.classList.add('busy'));
  send({action:'query',query:value});
}

/* ── Scroll + highlight section (works inside Streamlit iframe) ─────── */
function scrollToSection(sectionId){
  const el=document.getElementById(sectionId);
  if(!el)return;
  el.classList.add('rotating-highlight');
  clearTimeout(el._highlightTimer);
  el._highlightTimer=setTimeout(()=>el.classList.remove('rotating-highlight'),4000);

  // Streamlit's visible page is usually scrolled by an app-view container,
  // not by window/document. Scroll that real container, using the iframe's
  // position plus the section's position inside this component.
  try{
    const parent=window.parent;
    const frame=window.frameElement;
    if(!frame || !parent || parent===window) return;
    const doc=parent.document;
    const candidates=[
      doc.querySelector('[data-testid="stAppViewContainer"]'),
      doc.querySelector('[data-testid="stMain"]'),
      doc.querySelector('section.main'),
      doc.scrollingElement,
      doc.documentElement
    ].filter(Boolean);
    let scroller=candidates.find(node=>node.scrollHeight>node.clientHeight+4 &&
      (parent.getComputedStyle(node).overflowY==='auto'||parent.getComputedStyle(node).overflowY==='scroll'||node===doc.scrollingElement||node===doc.documentElement));
    if(!scroller) scroller=candidates[0];
    const frameRect=frame.getBoundingClientRect();
    const elRect=el.getBoundingClientRect();
    const scrollRect=scroller===doc.scrollingElement||scroller===doc.documentElement
      ? {top:0}:scroller.getBoundingClientRect();
    const currentTop=scroller.scrollTop||0;
    const target=currentTop+frameRect.top-scrollRect.top+elRect.top-24;
    if(typeof scroller.scrollTo==='function') scroller.scrollTo({top:Math.max(0,target),behavior:'smooth'});
    else scroller.scrollTop=Math.max(0,target);
  }catch(e){
    // Never scroll the iframe itself: doing so can trap the user's wheel and
    // prevent them from returning to the top of the Streamlit page.
  }
}

/* ── Sidebar HTML ────────────────────────────────────────────────────── */
function sidebar(){
  const nav=args.nav||'home';
  const collapsed = !!args.sidebar_collapsed;
  return `<aside class="sidebar ${collapsed ? 'sidebar-collapsed' : ''}">
    <button class="eda-nav-toggle" data-sidebar-toggle title="${collapsed ? 'Expand navigation' : 'Collapse navigation'}" aria-label="${collapsed ? 'Expand navigation' : 'Collapse navigation'}">${collapsed ? '»' : '«'}</button>
    <div class="brand"><div class="mark">A</div><div><div class="brandname">AUTOLYTICS</div><div class="tagline">From Data to Decisions</div></div></div>
    ${navItem('Home','home','⌂')}
    ${navItem('Upload Dataset','upload','⇧')}
    ${navItem('Query Dataset','query','⌕')}
    ${navItem('EDA &amp; Charts','eda','▥')}
    ${navItem('Modeling','modeling','♧')}
    ${navItem('Reports','reports','▤')}
    <div class="quote">"Better data.<br>Smarter decisions."<br><span style="font-size:22px">—</span></div>
  </aside>`;
}

/* ── EDA shell (compact — Python renders charts below iframe) ────────── */
function renderEda(){
  // Keep the custom AUTOLYTICS navigation visible while the Streamlit-native
  // EDA dashboard renders in the parent page. The iframe is fixed to the
  // sidebar width by app.py, so this shell must not render duplicate content.
  const collapsed = !!args.sidebar_collapsed;
  return `<div class="layout eda-mode ${collapsed ? 'is-collapsed' : ''}">${sidebar()}<main class="main" aria-hidden="true"></main></div>`;
}

/* ── Home page (upload + dataset preview + query — all on one page) ──── */
function renderHome(){
  const nav=args.nav||'home';
  const data=args.data||{};
  const rows=data.rows||[],cols=data.columns||[],total=data.total_rows||0;
  const file=args.file||null;

  let html=`<div class="layout ${args.sidebar_collapsed ? 'is-collapsed' : ''}">${sidebar()}<main class="main">`;

  /* Hero / Upload section — always visible on home page */
  const uploadHighlight=nav==='upload'?'rotating-highlight':'';
  html+=`<section class="hero">
    <div class="hero-copy">
      <div class="eyebrow">WELCOME TO AUTOLYTICS</div>
      <h1>Upload your dataset and<br>start exploring your data</h1>
      <p class="sub">Get instant insights with automatic cleaning, smart queries and data analysis.</p>
      <div class="art"><span>☁</span><span style="font-size:34px">▤</span><span style="font-size:12px;font-weight:800">CSV&nbsp;&nbsp; XLSX</span></div>
    </div>
    <div class="filecard ${uploadHighlight}" id="uploadSection">
      <div class="drop" id="drop">
        <div class="uploadicon">⇧</div>
        <strong>Drag &amp; drop your file here</strong>
        <small>or</small>
        <button class="run" id="chooseFile" type="button" style="margin-top:12px;min-height:37px">Choose File</button>
        <small>Supports CSV and Excel files (.csv, .xlsx)</small>
        <input id="file" type="file" accept=".csv,.xlsx" class="hidden">
      </div>
      ${file?`<div class="fileinfo"><span style="font-size:24px;color:#8aa6db">▤</span><span class="fname">${esc(file.name)}</span><span>${esc(file.size_label)}</span><button data-action="clearfile" title="Remove file">×</button></div>`:''}
    </div>
  </section>`;

  /* Dataset preview — visible once a file is loaded */
  if(rows.length||total){
    html+=`<section class="card">
      <div class="cardhead">
        <div>
          <div class="headtitle"><span class="headicon">▤</span>Dataset Preview</div>
          <div class="muted" style="margin-top:6px">Automatically cleaned and imputed working dataset</div>
        </div>
        <div class="metrics">
          <div class="metric"><div class="mlabel">▦ Rows</div><div class="mvalue">${Number(total).toLocaleString()}</div></div>
          <div class="metric"><div class="mlabel">☷ Columns</div><div class="mvalue">${cols.length}</div></div>
          <div class="metric"><div class="mlabel">▢ File Size</div><div class="mvalue">${esc(file?.size_label||'—')}</div></div>
        </div>
      </div>
      ${args.cleaning_summary?`<div class="cleaning">✓ ${esc(args.cleaning_summary)}</div>`:''}
      <div class="tablewrap">
        <table class="table">
          <thead><tr>${cols.map(c=>`<th>${esc(c.name)}<span>(${typeName(c.type)})</span></th>`).join('')}</tr></thead>
          <tbody>${rows.map(r=>`<tr>${cols.map(c=>`<td>${fmt(r[c.name])}</td>`).join('')}</tr>`).join('')}</tbody>
        </table>
      </div>
      <div class="tablefoot">
        <span>Showing ${total?Math.min((state.page-1)*5+1,total):0}–${Math.min(state.page*5,total)} of ${Number(total).toLocaleString()} rows</span>
        <div class="pages">
          <button data-page="${Math.max(1,state.page-1)}">‹</button>
          <button class="active">${state.page}</button>
          <button data-page="${Math.min(Math.ceil(total/5),state.page+1)}">›</button>
        </div>
      </div>
    </section>`;
  }else{
    html+=`<section class="card" style="text-align:center;padding:25px;color:#7185a6">Your dataset preview will appear here after you upload a CSV or Excel file.</section>`;
  }

  /* Query section — always on the home page; highlighted when nav=query */
  const queryHighlight=nav==='query'?'rotating-highlight':'';
  const qResult=args.query_result;
  const qError=args.query_error;
  html+=`<section class="card ${queryHighlight}" id="querySection">
    <div class="querytitle"><span class="headicon">✦</span>Ask a Question About Your Data</div>
    <div class="qsub">Type your query in natural language or write SQL. Get instant results below.</div>
    <div class="queryrow">
      <textarea id="query" placeholder="Example: Show total sales by city  |  Which product has the highest revenue?  |  Write a SQL query...">${esc(state.query)}</textarea>
      <button class="run" id="run">➤ &nbsp;Run Query</button>
    </div>
    <div class="suggestions">
      <b>✦ &nbsp;Try asking:</b>
      <button class="chip" data-query="Show the first 5 rows">Show the first 5 rows</button>
      <button class="chip" data-query="Average Age by Gender">Average age by gender</button>
      <button class="chip" data-query="Count records by category">Count records by category</button>
      <button class="chip" data-query="How many records are there?">Count records</button>
    </div>
    ${qError?`<div class="msg">${esc(qError)}</div>`:''}
    ${qResult?`<div class="result">
      ${qResult.sql?`<div style="font-weight:750;font-size:12px;margin-bottom:8px">${qResult.generated_sql?'Generated SQL':'Executed SQL'}</div><div class="sql">${esc(qResult.sql)}</div>`:''}
      ${qResult.rows?`<div style="font-weight:750;font-size:12px;margin:13px 0 8px">Query Results</div>
        <div class="tablewrap"><table class="table">
          <thead><tr>${qResult.columns.map(c=>`<th>${esc(c)}</th>`).join('')}</tr></thead>
          <tbody>${qResult.rows.map(r=>`<tr>${qResult.columns.map(c=>`<td>${fmt(r[c])}</td>`).join('')}</tr>`).join('')}</tbody>
        </table></div>
        <div class="muted" style="margin-top:8px">${qResult.count} result rows</div>`:''}
    </div>`:''}
  </section>`;

  html+='</main></div>';
  return html;
}

/* ── Master render ───────────────────────────────────────────────────── */
function render(){
  // Reset busy flags when results arrive
  if(args.query_result||args.query_error)state.busyQuery=false;
  if(args.file)state.busyUpload=false;
  if(args.eda_has_result||args.eda_error)state.busyEda=false;

  const nav=args.nav||'home';

  if(nav==='eda'){
    // On the EDA page Python renders all content below this iframe.
    // Report height=1 so Streamlit allocates no space for the component.
    root.innerHTML=renderEda();
    bindCommon();
    window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:setFrameHeight',height:1},'*');
    return;
  }
  if(nav==='modeling'){
    // Modeling page: Python renders all content; iframe is sidebar-only via fixed CSS.
    root.innerHTML=`<div class="layout ${args.sidebar_collapsed ? 'is-collapsed' : ''}">${sidebar()}<main class="main"></main></div>`;
    bindCommon();
    window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:setFrameHeight',height:1},'*');
    return;
  }
  if(nav==='reports'){
    root.innerHTML=`<div class="layout ${args.sidebar_collapsed ? 'is-collapsed' : ''}">${sidebar()}<main class="main"></main></div>`;
    bindCommon();
    window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:setFrameHeight',height:1},'*');
    return;
  }
  // home / upload / query all render the same full home page
  root.innerHTML=renderHome();
  bindCommon();
  bindHome();
  // Report height after DOM paint so scrollHeight is accurate
  requestAnimationFrame(()=>{
    height();
    // Re-report after images/fonts settle
    setTimeout(height,400);
  });

  // Do not auto-scroll during ordinary rerenders (including successful uploads).
  // Section scrolling is triggered only by an explicit sidebar navigation click.
}

/* ── Event binding ───────────────────────────────────────────────────── */
function bindCommon(){
  document.querySelectorAll('[data-sidebar-toggle]').forEach(el=>{
    el.onclick=e=>{
      e.preventDefault();e.stopPropagation();
      args.sidebar_collapsed=!args.sidebar_collapsed;
      render();
      send({action:'sidebar_toggle'});
    };
  });
  // Nav clicks — one listener per element, no duplicates (innerHTML replaced each render)
  document.querySelectorAll('[data-nav]').forEach(el=>{
    el.onclick=e=>{
      e.preventDefault();e.stopPropagation();
      const target=el.dataset.nav;
      // For upload/query: update nav state and send to Python,
      // but keep on the home page by not changing args.nav locally —
      // let Python send back the updated nav on rerun.
      args.nav=target;
      render();
      send({action:'nav',nav:target});
      if(target==='upload'||target==='query'){
        const sectionId=target==='upload'?'uploadSection':'querySection';
        requestAnimationFrame(()=>scrollToSection(sectionId));
        setTimeout(()=>scrollToSection(sectionId),180);
      }
    };
  });
}

function bindHome(){
  // File input
  const inp=document.getElementById('file');
  const drop=document.getElementById('drop');
  if(inp&&drop){
    drop.onclick=e=>{if(!e.target.closest('#chooseFile'))inp.click();};
    const choose=document.getElementById('chooseFile');
    if(choose)choose.onclick=e=>{e.preventDefault();e.stopPropagation();inp.click();};
    inp.onchange=()=>{if(inp.files[0])readFile(inp.files[0]);};
    drop.ondragover=e=>{e.preventDefault();drop.style.background='#edf4ff';};
    drop.ondragleave=()=>{drop.style.background='';};
    drop.ondrop=e=>{e.preventDefault();drop.style.background='';if(e.dataTransfer.files[0])readFile(e.dataTransfer.files[0]);};
  }
  // Clear file
  document.querySelectorAll('[data-action="clearfile"]').forEach(el=>{
    el.onclick=()=>send({action:'clear_file'});
  });
  // Pagination
  document.querySelectorAll('[data-page]').forEach(el=>{
    el.onclick=()=>{state.page=Number(el.dataset.page);send({action:'page',page:state.page});};
  });
  // Query textarea
  const q=document.getElementById('query');
  if(q){
    q.oninput=()=>{state.query=q.value;};
    q.onkeydown=e=>{
      if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();e.stopPropagation();sendQuery(q.value);}
    };
  }
  // Run button
  const run=document.getElementById('run');
  if(run)run.onclick=e=>{e.preventDefault();sendQuery(q?q.value:'');};
  // Suggestion chips
  document.querySelectorAll('[data-query]').forEach(el=>{
    el.onclick=()=>{state.query=el.dataset.query;sendQuery(state.query);};
  });
}

/* ── File reader + upload ────────────────────────────────────────────── */
async function blobToBase64(blob){
  return new Promise((resolve,reject)=>{
    const reader=new FileReader();
    reader.onload=()=>{const r=String(reader.result||'');resolve(r.includes(',')?r.slice(r.indexOf(',')+1):r);};
    reader.onerror=()=>reject(new Error('Could not encode file'));
    reader.readAsDataURL(blob);
  });
}
async function readFile(file){
  if(state.busyUpload)return;
  const name=file.name.toLowerCase();
  if(!name.endsWith('.csv')&&!name.endsWith('.xlsx')){send({action:'upload_error',error:'Please choose a CSV or XLSX file.'});return;}
  state.busyUpload=true;
  uploadStatus('Preparing your dataset…',file.size>15*1024*1024?'Large file detected. Compressing to speed up transfer.':'Reading selected file.');
  await new Promise(resolve=>requestAnimationFrame(()=>resolve()));
  try{
    let payloadBlob=file,compressed=false;
    if(name.endsWith('.csv')&&'CompressionStream' in window){
      uploadStatus('Compressing CSV…','This reduces upload time for large CSV files.');
      payloadBlob=await new Response(file.stream().pipeThrough(new CompressionStream('gzip'))).blob();
      compressed=true;
    }
    uploadStatus('Sending dataset…','Please keep this tab open while AUTOLYTICS processes the file.');
    const data=await blobToBase64(payloadBlob);
    send({action:'upload',name:file.name,size:file.size,data,compressed});
  }catch(err){
    state.busyUpload=false;
    send({action:'upload_error',error:'The selected file could not be read. Please try again.'});
  }
}

/* ── Streamlit message handler ───────────────────────────────────────── */
window.addEventListener('message',e=>{
  if(!e.data||e.data.type!=='streamlit:render')return;
  args=e.data.args||{};
  if(args.query_result||args.query_error)state.busyQuery=false;
  if(args.file)state.busyUpload=false;
  if(args.eda_has_result||args.eda_error)state.busyEda=false;
  render();
  // Intentionally do not restore scroll position on Streamlit reruns.
  // Navigation clicks perform one explicit scroll; reruns must never pull
  // the user back down after they manually scroll upward.
});
// Send ready and an initial height so Streamlit pre-allocates space before
// the first streamlit:render message arrives. Without this the component
// wrapper collapses to 0 and the content never paints on Cloud.
ready();
requestAnimationFrame(height);
