
var ortWasmThreaded = (() => {
  var _scriptDir = typeof document !== 'undefined' && document.currentScript ? document.currentScript.src : undefined;
  if (typeof __filename !== 'undefined') _scriptDir = _scriptDir || __filename;
  return (
function(ortWasmThreaded = {})  {

function aa(){c.buffer!=l.buffer&&m();return l}function n(){c.buffer!=l.buffer&&m();return ba}function q(){c.buffer!=l.buffer&&m();return ca}function u(){c.buffer!=l.buffer&&m();return da}function ea(){c.buffer!=l.buffer&&m();return fa}var w;w||(w=typeof ortWasmThreaded !== 'undefined' ? ortWasmThreaded : {});var ha,y;w.ready=new Promise((a,b)=>{ha=a;y=b});
var ia=Object.assign({},w),ja="./this.program",z=(a,b)=>{throw b;},ka="object"==typeof window,A="function"==typeof importScripts,B="object"==typeof process&&"object"==typeof process.versions&&"string"==typeof process.versions.node,D=w.ENVIRONMENT_IS_PTHREAD||!1,E="";function la(a){return w.locateFile?w.locateFile(a,E):E+a}var ma,F,H;
if(B){var fs=require("fs"),na=require("path");E=A?na.dirname(E)+"/":__dirname+"/";ma=(b,d)=>{b=b.startsWith("file://")?new URL(b):na.normalize(b);return fs.readFileSync(b,d?void 0:"utf8")};H=b=>{b=ma(b,!0);b.buffer||(b=new Uint8Array(b));return b};F=(b,d,f,h=!0)=>{b=b.startsWith("file://")?new URL(b):na.normalize(b);fs.readFile(b,h?void 0:"utf8",(g,k)=>{g?f(g):d(h?k.buffer:k)})};!w.thisProgram&&1<process.argv.length&&(ja=process.argv[1].replace(/\\/g,"/"));process.argv.slice(2);z=(b,d)=>{process.exitCode=
b;throw d;};w.inspect=()=>"[Emscripten Module object]";let a;try{a=require("worker_threads")}catch(b){throw console.error('The "worker_threads" module is not supported in this node.js build - perhaps a newer version is needed?'),b;}global.Worker=a.Worker}else if(ka||A)A?E=self.location.href:"undefined"!=typeof document&&document.currentScript&&(E=document.currentScript.src),_scriptDir&&(E=_scriptDir),0!==E.indexOf("blob:")?E=E.substr(0,E.replace(/[?#].*/,"").lastIndexOf("/")+1):E="",B||(ma=a=>{var b=
new XMLHttpRequest;b.open("GET",a,!1);b.send(null);return b.responseText},A&&(H=a=>{var b=new XMLHttpRequest;b.open("GET",a,!1);b.responseType="arraybuffer";b.send(null);return new Uint8Array(b.response)}),F=(a,b,d)=>{var f=new XMLHttpRequest;f.open("GET",a,!0);f.responseType="arraybuffer";f.onload=()=>{200==f.status||0==f.status&&f.response?b(f.response):d()};f.onerror=d;f.send(null)});B&&"undefined"==typeof performance&&(global.performance=require("perf_hooks").performance);
var oa=console.log.bind(console),pa=console.warn.bind(console);B&&(oa=(...a)=>fs.writeSync(1,a.join(" ")+"\n"),pa=(...a)=>fs.writeSync(2,a.join(" ")+"\n"));var qa=w.print||oa,I=w.printErr||pa;Object.assign(w,ia);ia=null;w.thisProgram&&(ja=w.thisProgram);w.quit&&(z=w.quit);var J;w.wasmBinary&&(J=w.wasmBinary);var noExitRuntime=w.noExitRuntime||!0;"object"!=typeof WebAssembly&&K("no native wasm support detected");var c,ra,L=!1,M,l,ba,ca,da,fa;
function m(){var a=c.buffer;w.HEAP8=l=new Int8Array(a);w.HEAP16=new Int16Array(a);w.HEAP32=ca=new Int32Array(a);w.HEAPU8=ba=new Uint8Array(a);w.HEAPU16=new Uint16Array(a);w.HEAPU32=da=new Uint32Array(a);w.HEAPF32=new Float32Array(a);w.HEAPF64=fa=new Float64Array(a)}var N=w.INITIAL_MEMORY||16777216;5242880<=N||K("INITIAL_MEMORY should be larger than STACK_SIZE, was "+N+"! (STACK_SIZE=5242880)");
if(D)c=w.wasmMemory;else if(w.wasmMemory)c=w.wasmMemory;else if(c=new WebAssembly.Memory({initial:N/65536,maximum:65536,shared:!0}),!(c.buffer instanceof SharedArrayBuffer))throw I("requested a shared WebAssembly.Memory but the returned buffer is not a SharedArrayBuffer, indicating that while the browser has SharedArrayBuffer it does not have WebAssembly threads support - you may need to set a flag"),B&&I("(on node you may need: --experimental-wasm-threads --experimental-wasm-bulk-memory and/or recent version)"),
Error("bad memory");m();N=c.buffer.byteLength;var sa,ta=[],ua=[],va=[],wa=0;function O(){return noExitRuntime||0<wa}function xa(){var a=w.preRun.shift();ta.unshift(a)}var P=0,ya=null,Q=null;function K(a){if(w.onAbort)w.onAbort(a);a="Aborted("+a+")";I(a);L=!0;M=1;a=new WebAssembly.RuntimeError(a+". Build with -sASSERTIONS for more info.");y(a);throw a;}function za(a){return a.startsWith("data:application/octet-stream;base64,")}var R;R="ort-wasm-simd-threaded.wasm";za(R)||(R=la(R));
function Aa(a){try{if(a==R&&J)return new Uint8Array(J);if(H)return H(a);throw"both async and sync fetching of the wasm failed";}catch(b){K(b)}}function Ba(a){if(!J&&(ka||A)){if("function"==typeof fetch&&!a.startsWith("file://"))return fetch(a,{credentials:"same-origin"}).then(b=>{if(!b.ok)throw"failed to load wasm binary file at '"+a+"'";return b.arrayBuffer()}).catch(()=>Aa(a));if(F)return new Promise((b,d)=>{F(a,f=>b(new Uint8Array(f)),d)})}return Promise.resolve().then(()=>Aa(a))}
function Ca(a,b,d){return Ba(a).then(f=>WebAssembly.instantiate(f,b)).then(f=>f).then(d,f=>{I("failed to asynchronously prepare wasm: "+f);K(f)})}
function Da(a,b){var d=R;return J||"function"!=typeof WebAssembly.instantiateStreaming||za(d)||d.startsWith("file://")||B||"function"!=typeof fetch?Ca(d,a,b):fetch(d,{credentials:"same-origin"}).then(f=>WebAssembly.instantiateStreaming(f,a).then(b,function(h){I("wasm streaming compile failed: "+h);I("falling back to ArrayBuffer instantiation");return Ca(d,a,b)}))}function Ea(a){this.name="ExitStatus";this.message="Program terminated with exit("+a+")";this.status=a}
function Fa(a){a.terminate();a.onmessage=()=>{}}function Ga(a){(a=S.Ha[a])||K();S.hb(a)}function Ha(a){var b=S.bb();if(!b)return 6;S.Ka.push(b);S.Ha[a.Ja]=b;b.Ja=a.Ja;var d={cmd:"run",start_routine:a.ib,arg:a.$a,pthread_ptr:a.Ja};B&&b.unref();b.postMessage(d,a.ob);return 0}var Ia="undefined"!=typeof TextDecoder?new TextDecoder("utf8"):void 0;
function Ja(a,b,d){b>>>=0;var f=b+d;for(d=b;a[d]&&!(d>=f);)++d;if(16<d-b&&a.buffer&&Ia)return Ia.decode(a.buffer instanceof SharedArrayBuffer?a.slice(b,d):a.subarray(b,d));for(f="";b<d;){var h=a[b++];if(h&128){var g=a[b++]&63;if(192==(h&224))f+=String.fromCharCode((h&31)<<6|g);else{var k=a[b++]&63;h=224==(h&240)?(h&15)<<12|g<<6|k:(h&7)<<18|g<<12|k<<6|a[b++]&63;65536>h?f+=String.fromCharCode(h):(h-=65536,f+=String.fromCharCode(55296|h>>10,56320|h&1023))}}else f+=String.fromCharCode(h)}return f}
function Ka(a,b){return(a>>>=0)?Ja(n(),a,b):""}function La(a){if(D)return T(1,1,a);M=a;if(!O()){S.jb();if(w.onExit)w.onExit(a);L=!0}z(a,new Ea(a))}function Ma(a){M=a;if(D)throw Na(a),"unwind";La(a)}function Oa(a){a instanceof Ea||"unwind"==a||z(1,a)}
var S={Na:[],Ka:[],Va:[],Ha:{},Ra:function(){D&&S.cb()},rb:function(){},cb:function(){S.receiveObjectTransfer=S.gb;S.threadInitTLS=S.Ua;S.setExitStatus=S.Ta;noExitRuntime=!1},Ta:function(a){M=a},tb:["$terminateWorker"],jb:function(){for(var a of S.Ka)Fa(a);for(a of S.Na)Fa(a);S.Na=[];S.Ka=[];S.Ha=[]},hb:function(a){var b=a.Ja;delete S.Ha[b];S.Na.push(a);S.Ka.splice(S.Ka.indexOf(a),1);a.Ja=0;Pa(b)},gb:function(){},Ua:function(){S.Va.forEach(a=>a())},fb:a=>new Promise(b=>{a.onmessage=g=>{g=g.data;var k=
g.cmd;a.Ja&&(S.ab=a.Ja);if(g.targetThread&&g.targetThread!=U()){var t=S.Ha[g.sb];t?t.postMessage(g,g.transferList):I('Internal error! Worker sent a message "'+k+'" to target pthread '+g.targetThread+", but that thread no longer exists!")}else if("checkMailbox"===k)V();else if("spawnThread"===k)Ha(g);else if("cleanupThread"===k)Ga(g.thread);else if("killThread"===k)g=g.thread,k=S.Ha[g],delete S.Ha[g],Fa(k),Pa(g),S.Ka.splice(S.Ka.indexOf(k),1),k.Ja=0;else if("cancelThread"===k)S.Ha[g.thread].postMessage({cmd:"cancel"});
else if("loaded"===k)a.loaded=!0,b(a);else if("print"===k)qa("Thread "+g.threadId+": "+g.text);else if("printErr"===k)I("Thread "+g.threadId+": "+g.text);else if("alert"===k)alert("Thread "+g.threadId+": "+g.text);else if("setimmediate"===g.target)a.postMessage(g);else if("callHandler"===k)w[g.handler](...g.args);else k&&I("worker sent an unknown command "+k);S.ab=void 0};a.onerror=g=>{I("worker sent an error! "+g.filename+":"+g.lineno+": "+g.message);throw g;};B&&(a.on("message",function(g){a.onmessage({data:g})}),
a.on("error",function(g){a.onerror(g)}));var d=[],f=["onExit","onAbort","print","printErr"],h;for(h of f)w.hasOwnProperty(h)&&d.push(h);a.postMessage({cmd:"load",handlers:d,urlOrBlob:w.mainScriptUrlOrBlob||_scriptDir,wasmMemory:c,wasmModule:ra})}),eb:function(a){a()},Za:function(){var a=la("ort-wasm-simd-threaded.worker.js");a=new Worker(a);S.Na.push(a)},bb:function(){0==S.Na.length&&(S.Za(),S.fb(S.Na[0]));return S.Na.pop()}};w.PThread=S;function W(a){for(;0<a.length;)a.shift()(w)}
w.establishStackSpace=function(){var a=U(),b=q()[a+52>>2>>>0];a=q()[a+56>>2>>>0];Qa(b,b-a);X(b)};function Na(a){if(D)return T(2,0,a);Ma(a)}var Y=[];w.invokeEntryPoint=function(a,b){var d=Y[a];d||(a>=Y.length&&(Y.length=a+1),Y[a]=d=sa.get(a));a=d(b);O()?S.Ta(a):Ra(a)};function Sa(a){this.Qa=a-24;this.Ya=function(b){u()[this.Qa+4>>2>>>0]=b};this.Xa=function(b){u()[this.Qa+8>>2>>>0]=b};this.Ra=function(b,d){this.Wa();this.Ya(b);this.Xa(d)};this.Wa=function(){u()[this.Qa+16>>2>>>0]=0}}var Ta=0,Ua=0;
function Va(a,b,d,f){return D?T(3,1,a,b,d,f):Wa(a,b,d,f)}function Wa(a,b,d,f){if("undefined"==typeof SharedArrayBuffer)return I("Current environment does not support SharedArrayBuffer, pthreads are not available!"),6;var h=[];if(D&&0===h.length)return Va(a,b,d,f);a={ib:d,Ja:a,$a:f,ob:h};return D?(a.qb="spawnThread",postMessage(a,h),0):Ha(a)}function Xa(a,b,d){return D?T(4,1,a,b,d):0}function Ya(a,b){if(D)return T(5,1,a,b)}
function Za(a){for(var b=0,d=0;d<a.length;++d){var f=a.charCodeAt(d);127>=f?b++:2047>=f?b+=2:55296<=f&&57343>=f?(b+=4,++d):b+=3}return b}
function $a(a,b,d,f){d>>>=0;if(!(0<f))return 0;var h=d;f=d+f-1;for(var g=0;g<a.length;++g){var k=a.charCodeAt(g);if(55296<=k&&57343>=k){var t=a.charCodeAt(++g);k=65536+((k&1023)<<10)|t&1023}if(127>=k){if(d>=f)break;b[d++>>>0]=k}else{if(2047>=k){if(d+1>=f)break;b[d++>>>0]=192|k>>6}else{if(65535>=k){if(d+2>=f)break;b[d++>>>0]=224|k>>12}else{if(d+3>=f)break;b[d++>>>0]=240|k>>18;b[d++>>>0]=128|k>>12&63}b[d++>>>0]=128|k>>6&63}b[d++>>>0]=128|k&63}}b[d>>>0]=0;return d-h}
function ab(a,b,d){return $a(a,n(),b,d)}function bb(a,b){if(D)return T(6,1,a,b)}function cb(a,b,d){if(D)return T(7,1,a,b,d)}function db(a,b,d){return D?T(8,1,a,b,d):0}function eb(a,b){if(D)return T(9,1,a,b)}function fb(a,b,d){if(D)return T(10,1,a,b,d)}function gb(a,b,d,f){if(D)return T(11,1,a,b,d,f)}function hb(a,b,d,f){if(D)return T(12,1,a,b,d,f)}function ib(a,b,d,f){if(D)return T(13,1,a,b,d,f)}function jb(a){if(D)return T(14,1,a)}function kb(a,b){if(D)return T(15,1,a,b)}
function lb(a,b,d){if(D)return T(16,1,a,b,d)}function mb(a){if(!L)try{if(a(),!O())try{D?Ra(M):Ma(M)}catch(b){Oa(b)}}catch(b){Oa(b)}}function nb(a){"function"===typeof Atomics.pb&&(Atomics.pb(q(),a>>2,a).value.then(V),a+=128,Atomics.store(q(),a>>2,1))}w.__emscripten_thread_mailbox_await=nb;function V(){var a=U();a&&(nb(a),mb(()=>ob()))}w.checkMailbox=V;function pb(a){return u()[a>>>2]+4294967296*q()[a+4>>>2]}function Z(a){return 0===a%4&&(0!==a%100||0===a%400)}
var qb=[0,31,60,91,121,152,182,213,244,274,305,335],rb=[0,31,59,90,120,151,181,212,243,273,304,334];function sb(a){return(Z(a.getFullYear())?qb:rb)[a.getMonth()]+a.getDate()-1}function tb(a,b,d,f,h,g,k){return D?T(17,1,a,b,d,f,h,g,k):-52}function ub(a,b,d,f,h,g){if(D)return T(18,1,a,b,d,f,h,g)}function vb(a){var b=Za(a)+1,d=wb(b);d&&ab(a,d,b);return d}var xb;xb=B?()=>{var a=process.hrtime();return 1E3*a[0]+a[1]/1E6}:()=>performance.timeOrigin+performance.now();
function yb(a){var b=zb();a=a();X(b);return a}function T(a,b){var d=arguments.length-2,f=arguments;return yb(()=>{for(var h=Ab(8*d),g=h>>3,k=0;k<d;k++){var t=f[2+k];ea()[g+k>>>0]=t}return Bb(a,d,h,b)})}var Cb=[],Db={};
function Eb(){if(!Fb){var a={USER:"web_user",LOGNAME:"web_user",PATH:"/",PWD:"/",HOME:"/home/web_user",LANG:("object"==typeof navigator&&navigator.languages&&navigator.languages[0]||"C").replace("-","_")+".UTF-8",_:ja||"./this.program"},b;for(b in Db)void 0===Db[b]?delete a[b]:a[b]=Db[b];var d=[];for(b in a)d.push(b+"="+a[b]);Fb=d}return Fb}var Fb;
function Gb(a,b){if(D)return T(19,1,a,b);var d=0;Eb().forEach(function(f,h){var g=b+d;h=u()[a+4*h>>2>>>0]=g;for(g=0;g<f.length;++g)aa()[h++>>0>>>0]=f.charCodeAt(g);aa()[h>>0>>>0]=0;d+=f.length+1});return 0}function Hb(a,b){if(D)return T(20,1,a,b);var d=Eb();u()[a>>2>>>0]=d.length;var f=0;d.forEach(function(h){f+=h.length+1});u()[b>>2>>>0]=f;return 0}function Lb(a){return D?T(21,1,a):52}function Mb(a,b,d,f){return D?T(22,1,a,b,d,f):52}function Nb(a,b,d,f,h){return D?T(23,1,a,b,d,f,h):70}
var Ob=[null,[],[]];function Pb(a,b,d,f){if(D)return T(24,1,a,b,d,f);for(var h=0,g=0;g<d;g++){var k=u()[b>>2>>>0],t=u()[b+4>>2>>>0];b+=8;for(var C=0;C<t;C++){var v=n()[k+C>>>0],x=Ob[a];0===v||10===v?((1===a?qa:I)(Ja(x,0)),x.length=0):x.push(v)}h+=t}u()[f>>2>>>0]=h;return 0}
function Qb(){if("object"==typeof crypto&&"function"==typeof crypto.getRandomValues)return d=>(d.set(crypto.getRandomValues(new Uint8Array(d.byteLength))),d);if(B)try{var a=require("crypto");if(a.randomFillSync)return d=>a.randomFillSync(d);var b=a.randomBytes;return d=>(d.set(b(d.byteLength)),d)}catch(d){}K("initRandomDevice")}function Rb(a){return(Rb=Qb())(a)}var Sb=[31,29,31,30,31,30,31,31,30,31,30,31],Tb=[31,28,31,30,31,30,31,31,30,31,30,31];
function Ub(a){var b=Array(Za(a)+1);$a(a,b,0,b.length);return b}function Vb(a,b){aa().set(a,b>>>0)}
function Wb(a,b,d,f){function h(e,p,r){for(e="number"==typeof e?e.toString():e||"";e.length<p;)e=r[0]+e;return e}function g(e,p){return h(e,p,"0")}function k(e,p){function r(Ib){return 0>Ib?-1:0<Ib?1:0}var G;0===(G=r(e.getFullYear()-p.getFullYear()))&&0===(G=r(e.getMonth()-p.getMonth()))&&(G=r(e.getDate()-p.getDate()));return G}function t(e){switch(e.getDay()){case 0:return new Date(e.getFullYear()-1,11,29);case 1:return e;case 2:return new Date(e.getFullYear(),0,3);case 3:return new Date(e.getFullYear(),
0,2);case 4:return new Date(e.getFullYear(),0,1);case 5:return new Date(e.getFullYear()-1,11,31);case 6:return new Date(e.getFullYear()-1,11,30)}}function C(e){var p=e.La;for(e=new Date((new Date(e.Ma+1900,0,1)).getTime());0<p;){var r=e.getMonth(),G=(Z(e.getFullYear())?Sb:Tb)[r];if(p>G-e.getDate())p-=G-e.getDate()+1,e.setDate(1),11>r?e.setMonth(r+1):(e.setMonth(0),e.setFullYear(e.getFullYear()+1));else{e.setDate(e.getDate()+p);break}}r=new Date(e.getFullYear()+1,0,4);p=t(new Date(e.getFullYear(),
0,4));r=t(r);return 0>=k(p,e)?0>=k(r,e)?e.getFullYear()+1:e.getFullYear():e.getFullYear()-1}var v=q()[f+40>>2>>>0];f={mb:q()[f>>2>>>0],lb:q()[f+4>>2>>>0],Oa:q()[f+8>>2>>>0],Sa:q()[f+12>>2>>>0],Pa:q()[f+16>>2>>>0],Ma:q()[f+20>>2>>>0],Ia:q()[f+24>>2>>>0],La:q()[f+28>>2>>>0],ub:q()[f+32>>2>>>0],kb:q()[f+36>>2>>>0],nb:v?Ka(v):""};d=Ka(d);v={"%c":"%a %b %d %H:%M:%S %Y","%D":"%m/%d/%y","%F":"%Y-%m-%d","%h":"%b","%r":"%I:%M:%S %p","%R":"%H:%M","%T":"%H:%M:%S","%x":"%m/%d/%y","%X":"%H:%M:%S","%Ec":"%c","%EC":"%C",
"%Ex":"%m/%d/%y","%EX":"%H:%M:%S","%Ey":"%y","%EY":"%Y","%Od":"%d","%Oe":"%e","%OH":"%H","%OI":"%I","%Om":"%m","%OM":"%M","%OS":"%S","%Ou":"%u","%OU":"%U","%OV":"%V","%Ow":"%w","%OW":"%W","%Oy":"%y"};for(var x in v)d=d.replace(new RegExp(x,"g"),v[x]);var Jb="Sunday Monday Tuesday Wednesday Thursday Friday Saturday".split(" "),Kb="January February March April May June July August September October November December".split(" ");v={"%a":function(e){return Jb[e.Ia].substring(0,3)},"%A":function(e){return Jb[e.Ia]},
"%b":function(e){return Kb[e.Pa].substring(0,3)},"%B":function(e){return Kb[e.Pa]},"%C":function(e){return g((e.Ma+1900)/100|0,2)},"%d":function(e){return g(e.Sa,2)},"%e":function(e){return h(e.Sa,2," ")},"%g":function(e){return C(e).toString().substring(2)},"%G":function(e){return C(e)},"%H":function(e){return g(e.Oa,2)},"%I":function(e){e=e.Oa;0==e?e=12:12<e&&(e-=12);return g(e,2)},"%j":function(e){for(var p=0,r=0;r<=e.Pa-1;p+=(Z(e.Ma+1900)?Sb:Tb)[r++]);return g(e.Sa+p,3)},"%m":function(e){return g(e.Pa+
1,2)},"%M":function(e){return g(e.lb,2)},"%n":function(){return"\n"},"%p":function(e){return 0<=e.Oa&&12>e.Oa?"AM":"PM"},"%S":function(e){return g(e.mb,2)},"%t":function(){return"\t"},"%u":function(e){return e.Ia||7},"%U":function(e){return g(Math.floor((e.La+7-e.Ia)/7),2)},"%V":function(e){var p=Math.floor((e.La+7-(e.Ia+6)%7)/7);2>=(e.Ia+371-e.La-2)%7&&p++;if(p)53==p&&(r=(e.Ia+371-e.La)%7,4==r||3==r&&Z(e.Ma)||(p=1));else{p=52;var r=(e.Ia+7-e.La-1)%7;(4==r||5==r&&Z(e.Ma%400-1))&&p++}return g(p,2)},
"%w":function(e){return e.Ia},"%W":function(e){return g(Math.floor((e.La+7-(e.Ia+6)%7)/7),2)},"%y":function(e){return(e.Ma+1900).toString().substring(2)},"%Y":function(e){return e.Ma+1900},"%z":function(e){e=e.kb;var p=0<=e;e=Math.abs(e)/60;return(p?"+":"-")+String("0000"+(e/60*100+e%60)).slice(-4)},"%Z":function(e){return e.nb},"%%":function(){return"%"}};d=d.replace(/%%/g,"\x00\x00");for(x in v)d.includes(x)&&(d=d.replace(new RegExp(x,"g"),v[x](f)));d=d.replace(/\0\0/g,"%");x=Ub(d);if(x.length>
b)return 0;Vb(x,a);return x.length-1}S.Ra();
var Xb=[null,La,Na,Va,Xa,Ya,bb,cb,db,eb,fb,gb,hb,ib,jb,kb,lb,tb,ub,Gb,Hb,Lb,Mb,Nb,Pb],Zb={b:function(a,b,d){(new Sa(a)).Ra(b,d);Ta=a;Ua++;throw Ta;},p:function(){K("To use dlopen, you need enable dynamic linking, see https://emscripten.org/docs/compiling/Dynamic-Linking.html")},N:function(a){Yb(a,!A,1,!ka);S.Ua()},k:function(a){D?postMessage({cmd:"cleanupThread",thread:a}):Ga(a)},C:Wa,h:Xa,U:Ya,y:bb,B:cb,V:db,R:eb,J:fb,Q:gb,o:hb,A:ib,w:jb,T:kb,x:lb,D:function(){return 131072},Y:function(){return!0},
u:function(a,b){a==b?setTimeout(()=>V()):D?postMessage({targetThread:a,cmd:"checkMailbox"}):(a=S.Ha[a])&&a.postMessage({cmd:"checkMailbox"})},L:function(){return-1},M:nb,X:function(a){B&&S.Ha[a].ref()},G:function(a,b){a=new Date(1E3*pb(a));q()[b>>2>>>0]=a.getUTCSeconds();q()[b+4>>2>>>0]=a.getUTCMinutes();q()[b+8>>2>>>0]=a.getUTCHours();q()[b+12>>2>>>0]=a.getUTCDate();q()[b+16>>2>>>0]=a.getUTCMonth();q()[b+20>>2>>>0]=a.getUTCFullYear()-1900;q()[b+24>>2>>>0]=a.getUTCDay();a=(a.getTime()-Date.UTC(a.getUTCFullYear(),
0,1,0,0,0,0))/864E5|0;q()[b+28>>2>>>0]=a},H:function(a,b){a=new Date(1E3*pb(a));q()[b>>2>>>0]=a.getSeconds();q()[b+4>>2>>>0]=a.getMinutes();q()[b+8>>2>>>0]=a.getHours();q()[b+12>>2>>>0]=a.getDate();q()[b+16>>2>>>0]=a.getMonth();q()[b+20>>2>>>0]=a.getFullYear()-1900;q()[b+24>>2>>>0]=a.getDay();var d=sb(a)|0;q()[b+28>>2>>>0]=d;q()[b+36>>2>>>0]=-(60*a.getTimezoneOffset());d=(new Date(a.getFullYear(),6,1)).getTimezoneOffset();var f=(new Date(a.getFullYear(),0,1)).getTimezoneOffset();a=(d!=f&&a.getTimezoneOffset()==
Math.min(f,d))|0;q()[b+32>>2>>>0]=a},I:function(a){var b=new Date(q()[a+20>>2>>>0]+1900,q()[a+16>>2>>>0],q()[a+12>>2>>>0],q()[a+8>>2>>>0],q()[a+4>>2>>>0],q()[a>>2>>>0],0),d=q()[a+32>>2>>>0],f=b.getTimezoneOffset(),h=(new Date(b.getFullYear(),6,1)).getTimezoneOffset(),g=(new Date(b.getFullYear(),0,1)).getTimezoneOffset(),k=Math.min(g,h);0>d?q()[a+32>>2>>>0]=Number(h!=g&&k==f):0<d!=(k==f)&&(h=Math.max(g,h),b.setTime(b.getTime()+6E4*((0<d?k:h)-f)));q()[a+24>>2>>>0]=b.getDay();d=sb(b)|0;q()[a+28>>2>>>
0]=d;q()[a>>2>>>0]=b.getSeconds();q()[a+4>>2>>>0]=b.getMinutes();q()[a+8>>2>>>0]=b.getHours();q()[a+12>>2>>>0]=b.getDate();q()[a+16>>2>>>0]=b.getMonth();q()[a+20>>2>>>0]=b.getYear();return b.getTime()/1E3|0},E:tb,F:ub,t:function(a,b,d){function f(v){return(v=v.toTimeString().match(/\(([A-Za-z ]+)\)$/))?v[1]:"GMT"}var h=(new Date).getFullYear(),g=new Date(h,0,1),k=new Date(h,6,1);h=g.getTimezoneOffset();var t=k.getTimezoneOffset(),C=Math.max(h,t);u()[a>>2>>>0]=60*C;q()[b>>2>>>0]=Number(h!=t);a=f(g);
b=f(k);a=vb(a);b=vb(b);t<h?(u()[d>>2>>>0]=a,u()[d+4>>2>>>0]=b):(u()[d>>2>>>0]=b,u()[d+4>>2>>>0]=a)},c:function(){K("")},S:function(){K("To use dlopen, you need enable dynamic linking, see https://emscripten.org/docs/compiling/Dynamic-Linking.html")},l:function(){},i:function(){return Date.now()},W:function(){wa+=1;throw"unwind";},v:function(){return 4294901760},e:xb,f:function(){return B?require("os").cpus().length:navigator.hardwareConcurrency},K:function(a,b,d){Cb.length=b;d>>=3;for(var f=0;f<b;f++)Cb[f]=
ea()[d+f>>>0];return Xb[a].apply(null,Cb)},s:function(a){var b=n().length;a>>>=0;if(a<=b||4294901760<a)return!1;for(var d=1;4>=d;d*=2){var f=b*(1+.2/d);f=Math.min(f,a+100663296);var h=Math,g=h.min;f=Math.max(a,f);f+=(65536-f%65536)%65536;a:{var k=c.buffer;try{c.grow(g.call(h,4294901760,f)-k.byteLength+65535>>>16);m();var t=1;break a}catch(C){}t=void 0}if(t)return!0}return!1},O:Gb,P:Hb,j:Ma,g:Lb,n:Mb,q:Nb,m:Pb,r:function(a,b){Rb(n().subarray(a>>>0,a+b>>>0));return 0},a:c||w.wasmMemory,z:Wb,d:function(a,
b,d,f){return Wb(a,b,d,f)}};
(function(){function a(d,f){d=d.exports;w.asm=d;S.Va.push(w.asm.va);sa=w.asm.wa;ua.unshift(w.asm.Z);ra=f;S.eb(()=>{P--;w.monitorRunDependencies&&w.monitorRunDependencies(P);if(0==P&&(null!==ya&&(clearInterval(ya),ya=null),Q)){var h=Q;Q=null;h()}});return d}var b={a:Zb};P++;w.monitorRunDependencies&&w.monitorRunDependencies(P);if(w.instantiateWasm)try{return w.instantiateWasm(b,a)}catch(d){I("Module.instantiateWasm callback failed with error: "+d),y(d)}Da(b,function(d){a(d.instance,d.module)}).catch(y);
return{}})();w._OrtInit=function(){return(w._OrtInit=w.asm._).apply(null,arguments)};w._OrtCreateSessionOptions=function(){return(w._OrtCreateSessionOptions=w.asm.$).apply(null,arguments)};w._OrtAppendExecutionProvider=function(){return(w._OrtAppendExecutionProvider=w.asm.aa).apply(null,arguments)};w._OrtAddSessionConfigEntry=function(){return(w._OrtAddSessionConfigEntry=w.asm.ba).apply(null,arguments)};
w._OrtReleaseSessionOptions=function(){return(w._OrtReleaseSessionOptions=w.asm.ca).apply(null,arguments)};w._OrtCreateSession=function(){return(w._OrtCreateSession=w.asm.da).apply(null,arguments)};w._OrtReleaseSession=function(){return(w._OrtReleaseSession=w.asm.ea).apply(null,arguments)};w._OrtGetInputCount=function(){return(w._OrtGetInputCount=w.asm.fa).apply(null,arguments)};w._OrtGetOutputCount=function(){return(w._OrtGetOutputCount=w.asm.ga).apply(null,arguments)};
w._OrtGetInputName=function(){return(w._OrtGetInputName=w.asm.ha).apply(null,arguments)};w._OrtGetOutputName=function(){return(w._OrtGetOutputName=w.asm.ia).apply(null,arguments)};w._OrtFree=function(){return(w._OrtFree=w.asm.ja).apply(null,arguments)};w._OrtCreateTensor=function(){return(w._OrtCreateTensor=w.asm.ka).apply(null,arguments)};w._OrtGetTensorData=function(){return(w._OrtGetTensorData=w.asm.la).apply(null,arguments)};
w._OrtReleaseTensor=function(){return(w._OrtReleaseTensor=w.asm.ma).apply(null,arguments)};w._OrtCreateRunOptions=function(){return(w._OrtCreateRunOptions=w.asm.na).apply(null,arguments)};w._OrtAddRunConfigEntry=function(){return(w._OrtAddRunConfigEntry=w.asm.oa).apply(null,arguments)};w._OrtReleaseRunOptions=function(){return(w._OrtReleaseRunOptions=w.asm.pa).apply(null,arguments)};w._OrtRun=function(){return(w._OrtRun=w.asm.qa).apply(null,arguments)};
w._OrtEndProfiling=function(){return(w._OrtEndProfiling=w.asm.ra).apply(null,arguments)};var U=w._pthread_self=function(){return(U=w._pthread_self=w.asm.sa).apply(null,arguments)},wb=w._malloc=function(){return(wb=w._malloc=w.asm.ta).apply(null,arguments)};w._free=function(){return(w._free=w.asm.ua).apply(null,arguments)};w.__emscripten_tls_init=function(){return(w.__emscripten_tls_init=w.asm.va).apply(null,arguments)};
var Yb=w.__emscripten_thread_init=function(){return(Yb=w.__emscripten_thread_init=w.asm.xa).apply(null,arguments)};w.__emscripten_thread_crashed=function(){return(w.__emscripten_thread_crashed=w.asm.ya).apply(null,arguments)};function Bb(){return(Bb=w.asm.za).apply(null,arguments)}function Pa(){return(Pa=w.asm.Aa).apply(null,arguments)}
var Ra=w.__emscripten_thread_exit=function(){return(Ra=w.__emscripten_thread_exit=w.asm.Ba).apply(null,arguments)},ob=w.__emscripten_check_mailbox=function(){return(ob=w.__emscripten_check_mailbox=w.asm.Ca).apply(null,arguments)};function Qa(){return(Qa=w.asm.Da).apply(null,arguments)}function zb(){return(zb=w.asm.Ea).apply(null,arguments)}function X(){return(X=w.asm.Fa).apply(null,arguments)}function Ab(){return(Ab=w.asm.Ga).apply(null,arguments)}w.keepRuntimeAlive=O;w.wasmMemory=c;
w.stackAlloc=Ab;w.stackSave=zb;w.stackRestore=X;w.UTF8ToString=Ka;w.stringToUTF8=ab;w.lengthBytesUTF8=Za;w.ExitStatus=Ea;w.PThread=S;var $b;Q=function ac(){$b||bc();$b||(Q=ac)};
function bc(){function a(){if(!$b&&($b=!0,w.calledRun=!0,!L)){D||W(ua);ha(w);if(w.onRuntimeInitialized)w.onRuntimeInitialized();if(!D){if(w.postRun)for("function"==typeof w.postRun&&(w.postRun=[w.postRun]);w.postRun.length;){var b=w.postRun.shift();va.unshift(b)}W(va)}}}if(!(0<P))if(D)ha(w),D||W(ua),startWorker(w);else{if(w.preRun)for("function"==typeof w.preRun&&(w.preRun=[w.preRun]);w.preRun.length;)xa();W(ta);0<P||(w.setStatus?(w.setStatus("Running..."),setTimeout(function(){setTimeout(function(){w.setStatus("")},
1);a()},1)):a())}}if(w.preInit)for("function"==typeof w.preInit&&(w.preInit=[w.preInit]);0<w.preInit.length;)w.preInit.pop()();bc();


  return ortWasmThreaded.ready
}

);
})();
if (typeof exports === 'object' && typeof module === 'object')
  module.exports = ortWasmThreaded;
else if (typeof define === 'function' && define['amd'])
  define([], function() { return ortWasmThreaded; });
else if (typeof exports === 'object')
  exports["ortWasmThreaded"] = ortWasmThreaded;
export default ortWasmThreaded;