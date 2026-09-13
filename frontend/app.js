const API = 'http://127.0.0.1:8000';
const STATES = [
  ['Arunachal Pradesh',27.1,93.6],['Assam',26.2,92.9],['Manipur',24.7,93.9],['Meghalaya',25.5,91.3],
  ['Mizoram',23.7,92.7],['Nagaland',26.0,94.5],['Sikkim',27.5,88.5],['Tripura',23.8,91.3]
];
let map, inventoryLayer, stateRiskLayer, selectedMarker;
const stateCache = {rows:null};
function $(id){return document.getElementById(id)}
function clock(){ if($('clock')) $('clock').textContent=new Date().toLocaleTimeString([], {hour12:false}); }
setInterval(clock,1000); clock();
function esc(v){return String(v??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m]));}
function riskClass(level){return level==='VERY HIGH'?'critical':level==='HIGH'?'high':level==='MODERATE'?'moderate':'low';}
function riskColor(level){return level==='VERY HIGH'?'#ff3f66':level==='HIGH'?'#ff7a2f':level==='MODERATE'?'#ffd21f':'#28e77f';}
function severityClass(level){return level==='CRITICAL'?'critical':level==='WARNING'?'high':level==='WATCH'?'moderate':level==='REPORT'?'report':'low';}
async function getJSON(path,opts={}){
  const r=await fetch(API+path,opts);
  if(!r.ok){let msg=await r.text();try{msg=JSON.parse(msg).detail||msg}catch{};throw new Error(msg);}
  return r.json();
}
function showToast(msg){const t=$('toast');if(!t)return;t.textContent=msg;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),3200);}

function nav(){
 document.querySelectorAll('.nav').forEach(btn=>btn.addEventListener('click',()=>{
  document.querySelectorAll('.nav').forEach(b=>b.classList.remove('active'));btn.classList.add('active');
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active-page'));$('page-'+btn.dataset.page).classList.add('active-page');
  if(btn.dataset.page==='map') setTimeout(initMap,30);
  if(btn.dataset.page==='weather') loadWeatherCards();
  if(btn.dataset.page==='alerts') loadAlerts();
  if(btn.dataset.page==='reports') loadReports();
  if(btn.dataset.page==='datasources') loadSources();
 }));
 $('openPrediction').onclick=()=>document.querySelector('[data-page="prediction"]').click();
 $('collapse').onclick=()=>document.querySelector('.sidebar').classList.toggle('collapsed');
}

async function loadModelStatus(){
 try{
  const d=await getJSON('/api/model/status');
  if($('modelBadge')){
   $('modelBadge').textContent=d.status==='READY' ? `● LIVE DATA • AI MODEL READY • ${d.algorithm||'validated research model'}` : '● LIVE DATA • AI MODEL NOT TRAINED';
  }
  if($('modelNotice') && d.status==='READY') $('modelNotice').textContent=`AI model active: ${d.algorithm}. Test ROC-AUC: ${d.test_metrics?.roc_auc ?? 'not available'}. The displayed AI value is a model risk score, not a calibrated probability.`;
 }catch(e){ if($('modelBadge')) $('modelBadge').textContent='● LIVE DATA • AI MODEL STATUS UNAVAILABLE'; }
}

async function loadStates(force=false){
 if(stateCache.rows && !force)return stateCache.rows;
 const data=await getJSON('/api/states');stateCache.rows=data.states;renderStateTable(stateCache.rows);return stateCache.rows;
}
function renderStateTable(rows){
 const body=$('stateTable');
 body.innerHTML=rows.map(x=>`<tr><td>${esc(x.name)}</td><td><span class="state-dot ${riskClass(x.risk_level||x.screening.level)}"></span>${esc(x.weather?.temperature_2m??'—')}°C</td><td>${esc(x.rainfall_24h??'—')} mm</td><td>${esc(x.risk_score??x.screening.score)}</td><td><span class="zone-tag ${riskClass(x.risk_level||x.screening.level)}">${esc(x.risk_level||x.screening.level)}</span></td></tr>`).join('');
 const nums=rows.map(x=>Number(x.risk_score??x.screening.score)).filter(Number.isFinite);
 const avg=nums.length?nums.reduce((a,b)=>a+b,0)/nums.length:NaN;
 $('overallScore').textContent=Number.isFinite(avg)?Math.round(avg):'—';
 const overall=rows.some(x=>(x.risk_level||x.screening.level)==='VERY HIGH')?'VERY HIGH':rows.some(x=>(x.risk_level||x.screening.level)==='HIGH')?'HIGH':rows.some(x=>(x.risk_level||x.screening.level)==='MODERATE')?'MODERATE':'LOW';
 $('overallLevel').textContent=overall;$('overallLevel').className='risk-badge '+riskClass(overall);
 $('zones').innerHTML=rows.filter(x=>['HIGH','VERY HIGH'].includes(x.risk_level||x.screening.level)).map(x=>`<div class="zone-row"><div><b>${esc(x.name)}</b><small>${esc(x.rainfall_24h??'—')} mm / 24h</small></div><span class="zone-tag ${riskClass(x.risk_level||x.screening.level)}">${esc(x.risk_level||x.screening.level)}</span></div>`).join('')||'<div class="small">No state currently reaches HIGH/VERY HIGH screening/model level.</div>';
 if(map) drawStateRisk(rows);
}
function drawStateRisk(rows){
 if(stateRiskLayer)stateRiskLayer.clearLayers();
 stateRiskLayer=L.layerGroup().addTo(map);
 rows.forEach(x=>{
   const level=x.risk_level||x.screening.level; const color=riskColor(level);
   L.circle([x.latitude,x.longitude],{radius:80000,color,weight:2,fillColor:color,fillOpacity:.16,opacity:.95}).addTo(stateRiskLayer)
     .bindPopup(`<b>${esc(x.name)}</b><br>Risk level: <strong>${esc(level)}</strong><br>Score: ${esc(x.risk_score??x.screening.score)}/100<br>24h rain: ${esc(x.rainfall_24h??'—')} mm<br>72h rain: ${esc(x.rainfall_72h??'—')} mm<br>${x.ai_prediction?`AI model: ${esc(x.ai_prediction.model_version)}`:'Screening only'}`);
   L.circleMarker([x.latitude,x.longitude],{radius:7,color:'#071014',weight:2,fillColor:color,fillOpacity:1}).addTo(stateRiskLayer).bindTooltip(`${x.name} — ${level}`);
 });
}
async function initMap(){
 if(map){map.invalidateSize();if(stateCache.rows)drawStateRisk(stateCache.rows);return;}
 map=L.map('map',{zoomControl:true}).setView([25.5788,91.8933],6);
 L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'© OpenStreetMap contributors'}).addTo(map);
 try{const rows=await loadStates();drawStateRisk(rows);}catch(e){$('inventoryCount').textContent='Weather source unavailable';}
 try{await loadInventory();}catch(e){$('inventoryCount').textContent='NASA GLC unavailable';}
 map.on('click',e=>analyzePoint(e.latlng.lat,e.latlng.lng));
}
async function loadInventory(force=false){
 const data=await getJSON(force?'/api/inventory?force=true':'/api/inventory');
 $('inventoryCount').textContent=`NASA GLC: ${data.count} historical events`;
 if($('inventoryStatus')){$('inventoryStatus').textContent=data.stale?'LOCAL CACHE':'CONNECTED';$('inventoryStatus').className=data.stale?'warn':'good';}
 if(inventoryLayer)inventoryLayer.clearLayers();inventoryLayer=L.layerGroup().addTo(map);
 (data.events||[]).forEach(ev=>L.circleMarker([ev.latitude,ev.longitude],{radius:4,color:'#ffe066',weight:1.2,fillColor:'#ffe066',fillOpacity:.9}).addTo(inventoryLayer).bindPopup(`<b>NASA historical landslide</b><br>${esc(ev.event_date||'Date unavailable')}<br>${esc(ev.landslide_type||'Type unavailable')}<br>${esc(ev.location||'Location unavailable')}`));
 return data;
}
async function refreshInventory(){try{await loadInventory(true);stateCache.rows=null;await loadStates(true);showToast('NASA inventory refreshed');}catch(e){showToast('NASA inventory unavailable');}}
async function locateAndAnalyze(){
 if(!navigator.geolocation){showToast('Browser geolocation is unavailable');return;}
 navigator.geolocation.getCurrentPosition(async p=>{
   const {latitude,longitude,accuracy}=p.coords;if(map){map.setView([latitude,longitude],10);} 
   ['lat','rlat'].forEach(id=>{if($(id))$(id).value=latitude.toFixed(6)});['lon','rlon','infraLon'].forEach(id=>{if($(id))$(id).value=longitude.toFixed(6)});$('infraLat').value=latitude.toFixed(6);
   if($('mapStatus'))$('mapStatus').textContent=`GPS selected • ±${Math.round(accuracy)} m`;
   try{await analyzePoint(latitude,longitude);}catch(e){showToast(e.message);}
 },()=>showToast('Location permission was denied or unavailable'),{enableHighAccuracy:true,timeout:15000,maximumAge:30000});
}
async function analyzePoint(lat,lon){
 if(selectedMarker)selectedMarker.remove(); selectedMarker=L.marker([lat,lon]).addTo(map);
 $('lat').value=lat.toFixed(6);$('lon').value=lon.toFixed(6);$('infraLat').value=lat.toFixed(6);$('infraLon').value=lon.toFixed(6);$('rlat').value=lat.toFixed(6);$('rlon').value=lon.toFixed(6);
 $('mapStatus').textContent='Loading ECMWF + SRTM + NASA data…';$('mapResult').innerHTML='<div class="loading">Fetching real weather, SRTM terrain and historical NASA events…</div>';
 const data=await getJSON('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({latitude:lat,longitude:lon,label:'Map selection'})});
 renderAnalysis(data,'mapResult');$('mapStatus').textContent=`${lat.toFixed(4)}, ${lon.toFixed(4)}`;
}
function renderAnalysis(data,target){
 const s=data.screening, f=data.final_assessment, ai=data.ai_prediction;
 const aiText=ai?`<div class="ai-panel"><b>AI model active</b><br>Algorithm: ${esc(ai.algorithm)}<br>Model risk score: <b>${esc(ai.score)}/100</b><br>Level: <b>${esc(ai.level)}</b><br><span class="small">${esc(ai.prediction_kind)}</span></div>`:'<div class="ai-panel muted"><b>AI model not active</b><br>No AI result is shown until a real model is trained and passes validation.</div>';
 $(target).innerHTML=`<div class="analysis-kpis"><div><span>${ai?'AI MODEL SCORE':'SCREENING SCORE'}</span><b class="${riskClass(f.level)}">${esc(f.score)}</b></div><div><span>LEVEL</span><b>${esc(f.level)}</b></div><div><span>24h RAIN</span><b>${esc(data.weather.rainfall_24h??'—')} mm</b></div><div><span>SLOPE</span><b>${esc(data.terrain.slope??'—')}°</b></div></div><div class="factor-list"><h4>Measured factors</h4>${s.reasons.map(r=>`<div>• ${esc(r)}</div>`).join('')||'<div>• No elevated weather/soil trigger detected by the screening layer.</div>'}<div>• SRTM elevation: <b>${esc(data.terrain.elevation??'—')} m</b></div><div>• SRTM slope: <b>${esc(data.terrain.slope??'—')}°</b></div><div>• NASA GLC events within 50 km: <b>${data.inventory.nearby_50km}</b></div><div>• NER NASA inventory records loaded: <b>${data.inventory.total_ner_events}</b></div><div>• GSI/NLFC: <b>Official portal verified</b></div><div>• Assessment basis: <b>${esc(f.basis)}</b></div>${aiText}</div><div class="small">Weather source: ${esc(data.weather.data_source)} · updated ${esc(data.weather.data_timestamp??'—')}</div>`;
 if($('modelNotice'))$('modelNotice').textContent=ai?`Real AI model active: ${ai.algorithm} (${ai.model_version}). Test ROC-AUC: ${data.model.test_metrics?.roc_auc ?? 'not available'}. The AI value is a model risk score, not a calibrated probability.`:'AI prediction is unavailable because no validated model is active. Weather screening remains available.';
}
async function runPrediction(){
 const payload={latitude:+$('lat').value,longitude:+$('lon').value,label:$('label').value};$('predScore').textContent='…';$('predScreenScore').textContent='…';
 try{
  const data=await getJSON('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  $('predScreenScore').textContent=data.screening.score;
  if(data.ai_prediction){$('predScore').textContent=data.ai_prediction.score;$('predLevel').textContent=data.ai_prediction.level;$('predLevel').className='risk-badge '+riskClass(data.ai_prediction.level);}
  else{$('predScore').textContent='N/A';$('predLevel').textContent='NOT ACTIVE';$('predLevel').className='risk-badge';}
  renderAnalysis(data,'predictionDetails');
 }catch(e){$('predictionDetails').textContent=e.message;$('predScore').textContent='—';$('predScreenScore').textContent='—';}
}
async function trainModel(){
 $('trainModelBtn').disabled=true;$('trainModelBtn').textContent='Building training data…';showToast('Building leakage-safe training data from NASA + historical weather + SRTM. This may take several minutes.');
 try{const d=await getJSON('/api/model/train',{method:'POST'}); if(d.status==='READY'){showToast(`AI model ready: ${d.algorithm}`);$('modelNotice').textContent=`Real model ready: ${d.algorithm}. Test ROC-AUC: ${d.test_metrics?.roc_auc ?? 'not available'}.`;await loadModelStatus();stateCache.rows=null;await loadStates(true);}else{showToast(d.message||d.status);$('modelNotice').textContent=d.message||d.status;}}
 catch(e){showToast('Model training failed');$('modelNotice').textContent=e.message;}
 finally{$('trainModelBtn').disabled=false;$('trainModelBtn').textContent='Train / update real AI model';}
}
async function loadWeatherCards(){
 const rows=await loadStates();$('weatherCards').innerHTML=rows.map(x=>`<div class="card weather-card"><div class="card-head"><b>${esc(x.name)}</b><span class="zone-tag ${riskClass(x.risk_level||x.screening.level)}">${esc(x.risk_level||x.screening.level)}</span></div><div class="weather-big">${esc(x.weather.temperature_2m??'—')}°C</div><div class="weather-stats"><span>1h<strong>${esc(x.rainfall_1h??'—')} mm</strong></span><span>24h<strong>${esc(x.rainfall_24h??'—')} mm</strong></span><span>72h<strong>${esc(x.rainfall_72h??'—')} mm</strong></span><span>soil<strong>${esc(x.soil_moisture_0_to_7cm??'—')}</strong></span></div><div class="small" style="margin-top:8px">${esc(x.weather_source)} · ${esc(x.model_run||'')}</div></div>`).join('');
}
async function loadInfrastructure(){
 try{const d=await getJSON(`/api/infrastructure?latitude=${+$('infraLat').value}&longitude=${+$('infraLon').value}&radius=${+$('infraRadius').value}`);$('infraResult').innerHTML=`<div class="analysis-kpis"><div><span>ROADS</span><b>${d.counts.roads}</b></div><div><span>AMENITIES</span><b>${d.counts.amenities}</b></div><div><span>BUILDINGS</span><b>${d.counts.buildings}</b></div></div><div class="small">Source: ${esc(d.source)} · ${esc(d.retrieved_at)}</div>`;}catch(e){$('infraResult').textContent='Infrastructure service unavailable.';}
}
async function loadAlerts(){
 try{
  const d=await getJSON('/api/alerts');
  const counts={CRITICAL:0,WARNING:0,WATCH:0,REPORT:0,INFO:0};(d.alerts||[]).forEach(a=>counts[a.severity]=(counts[a.severity]||0)+1);
  const parts=[];if(counts.CRITICAL)parts.push(`${counts.CRITICAL} CRITICAL`);if(counts.WARNING)parts.push(`${counts.WARNING} WARNING`);if(counts.WATCH)parts.push(`${counts.WATCH} WATCH`);if(counts.REPORT)parts.push(`${counts.REPORT} REPORT`);
  $('alertCount').textContent=parts.join(' · ')||'No active alerts';
  $('alertList').innerHTML=(d.alerts||[]).map(a=>`<div class="alert-row ${severityClass(a.severity)}"><div><b>${esc(a.title)}</b><small>${esc(a.message)}</small><small>Source: ${esc(a.source||'—')} · ${esc(a.created_at||'—')}</small></div><span class="zone-tag ${severityClass(a.severity)}">${esc(a.severity)}</span></div>`).join('') || '<div class="small">No active operational alerts. Monitoring continues.</div>';
  $('alertBasis').textContent=d.system_notice?`${d.system_notice.title}: ${d.system_notice.message} Official operational warnings should be taken from IMD/GSI.`:`Generated ${d.alerts.length} operational items from current monitoring data. These are not official emergency warnings.`;
 }catch(e){$('alertList').innerHTML=`<div class="small">Alert service unavailable: ${esc(e.message)}</div>`;$('alertCount').textContent='UNAVAILABLE';}
}
function loadReports(){getJSON('/api/reports').then(d=>$('savedReports').innerHTML=d.reports.map(r=>`<div class="report-item"><b>${esc(r.type)}</b><small>${esc(r.created_at)}</small><p>${esc(r.description)}</p><span>${esc(r.latitude)}, ${esc(r.longitude)}</span></div>`).join('')||'<div class="small">No saved reports.</div>').catch(()=>{});}
async function loadSources(){const d=await getJSON('/api/data-sources');$('sourceList').innerHTML=d.sources.map(s=>`<div class="source-row"><div><b>${esc(s.name)}</b><small>${esc(s.coverage)} · ${esc(s.details)}</small></div><span class="source-status ${String(s.status).toLowerCase().replaceAll(' ','-')}" title="${esc(s.last_checked||'')}">${esc(s.status)}</span></div>`).join('');}

$('analyzeBtn').onclick=runPrediction;$('infraBtn').onclick=loadInfrastructure;$('trainModelBtn').onclick=trainModel;$('locateBtn').onclick=locateAndAnalyze;$('refreshInventory').onclick=refreshInventory;
$('reportLocationBtn').onclick=()=>navigator.geolocation?.getCurrentPosition(p=>{$('rlat').value=p.coords.latitude.toFixed(6);$('rlon').value=p.coords.longitude.toFixed(6);$('lat').value=p.coords.latitude.toFixed(6);$('lon').value=p.coords.longitude.toFixed(6);if($('reportLocationStatus'))$('reportLocationStatus').textContent=`GPS location set • accuracy ±${Math.round(p.coords.accuracy)} m`;showToast('My location added to the report');},()=>showToast('Location permission was denied or unavailable'),{enableHighAccuracy:true,timeout:15000,maximumAge:30000});
$('reportForm').onsubmit=async e=>{e.preventDefault();const lat=+$('rlat').value,lon=+$('rlon').value;if(!Number.isFinite(lat)||!Number.isFinite(lon)){showToast('Please set a valid location or use My Location');return;}try{await getJSON('/api/reports',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({type:$('rtype').value,description:$('rdesc').value,latitude:lat,longitude:lon})});showToast('Report saved');e.target.reset();$('rlat').value='';$('rlon').value='';loadReports();}catch(err){showToast('Could not save report: '+err.message);}};

nav();
loadModelStatus();
loadStates().catch(()=>{});
