// Synthetic concurrent publication/communications drain boundary; no live services.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {Store, ROOT, COMMUNICATIONS_WORKER_LAP} from '../tools/daily_research/firestore_bridge.mjs';
import {MemoryFirestore} from './fixtures/daily_research/firestore-memory.mjs';

const NOW=Date.parse('2026-10-06T10:00:00Z');
const lap=(phase='active',until=NOW+180000)=>({schema_version:'blueprint.communications-worker-lap.v1',phase,
  lease:{owner:'communications-worker-lap:synthetic',generation:1,until},
  startedAt:NOW-1000,renewedAt:NOW-500,completedAt:phase==='complete'?NOW:null});
const deferred=()=>{let resolve;const promise=new Promise(done=>{resolve=done;});return {promise,resolve};};
function fixture(DB=MemoryFirestore) {
  const db=new DB();db.values.set(ROOT,{enabled:true,config:{synthetic:'preserved'}});
  return {db,store:new Store(db,()=>NOW,'synthetic-publication')};
}

// Firestore retries when any directly read document changes, including creation
// of a document that was absent. This double exposes both actual commit orders.
class ContendedFirestore extends MemoryFirestore {
  constructor(){super();this.revisions=new Map();this.pause=null;this.conflicts=0;}
  async runTransaction(fn,{maxAttempts=3}={}) {
    for(let attempt=0;attempt<maxAttempts;attempt++) {
      const values=structuredClone(this.values),revisions=new Map(this.revisions),reads=new Map(),writes=[];
      const result=await fn({get:async ref=>{
        assert.equal(writes.length,0,'transaction reads precede writes');
        reads.set(ref.path,revisions.get(ref.path) ?? 0);
        const value=values.get(ref.path),snap={exists:value!==undefined,data:()=>structuredClone(value)};
        if(this.pause?.path===ref.path) {
          const pause=this.pause;this.pause=null;pause.read.resolve();await pause.resume.promise;
        }
        return snap;
      },set:(ref,value,options)=>writes.push({ref,value:structuredClone(value),merge:options?.merge})});
      if([...reads].some(([path,revision])=>(this.revisions.get(path) ?? 0)!==revision)) {
        this.conflicts++;continue;
      }
      for(const {ref,value,merge} of writes) {
        this.values.set(ref.path,merge?{...this.values.get(ref.path),...value}:value);
        this.revisions.set(ref.path,(this.revisions.get(ref.path) ?? 0)+1);
      }
      return result;
    }
    throw new Error('synthetic_transaction_conflict_limit');
  }
  pauseRead(path){const read=deferred(),resume=deferred();this.pause={path,read,resume};return this.pause;}
}

// Consumer side of the existing Web communications lap claim protocol: read the
// canonical research control in the same transaction that writes workerLap.
async function claimLap(db) {
  return db.runTransaction(async tx=>{
    const control=(await tx.get(db.doc(ROOT))).data(),lease=control?.lease;
    if(typeof lease?.owner==='string' && lease.owner.startsWith('research-release:')) {
      assert.ok(Number.isSafeInteger(lease.expires_at_ms),'trusted release expiry must be valid');
      if(lease.expires_at_ms>NOW) return false;
    }
    const prior=(await tx.get(db.doc(COMMUNICATIONS_WORKER_LAP))).data();
    if(prior && (prior.phase!=='complete' || prior.lease?.until!==0)) return false;
    tx.set(db.doc(COMMUNICATIONS_WORKER_LAP),lap());return true;
  });
}

test('publication acquire reads absent lap in lease transaction and uses consumer-recognized owner',async()=>{
  const {db,store}=fixture(),reads=[];const original=db.runTransaction.bind(db);
  db.runTransaction=(fn,options)=>original(tx=>fn({...tx,get:ref=>{reads.push(ref.path);return tx.get(ref);}}),options);
  await store.acquire('research_release');
  assert.deepEqual(reads,[ROOT,COMMUNICATIONS_WORKER_LAP]);
  assert.equal(db.values.get(ROOT).lease.owner,'research-release:synthetic-publication');
  assert.equal(await claimLap(db),false);assert.equal(db.values.has(COMMUNICATIONS_WORKER_LAP),false);
  await store.release();assert.equal(await claimLap(db),true);
});

test('lap claim wins after publication read of missing lap: retry refuses lease and preserves active lap',async()=>{
  const {db,store}=fixture(ContendedFirestore),pause=db.pauseRead(COMMUNICATIONS_WORKER_LAP);
  const acquire=store.acquire('research_release');const rejected=assert.rejects(acquire,/communications_worker_lap_active/);
  await pause.read.promise;
  assert.equal(await claimLap(db),true);pause.resume.resolve();await rejected;
  assert.equal(db.conflicts,1);assert.equal(db.values.get(ROOT).lease,undefined);
  assert.equal(db.values.get(COMMUNICATIONS_WORKER_LAP).phase,'active');
});

test('publication lease wins after lap control read: consumer retry cannot create first lap',async()=>{
  const {db,store}=fixture(ContendedFirestore),pause=db.pauseRead(ROOT),claim=claimLap(db);
  await pause.read.promise;await store.acquire('research_release');pause.resume.resolve();
  assert.equal(await claim,false);assert.equal(db.conflicts,1);
  assert.equal(db.values.has(COMMUNICATIONS_WORKER_LAP),false);
  assert.equal(db.values.get(ROOT).lease.owner,'research-release:synthetic-publication');
});

for(const phase of ['active','draining','lost','unknown'])
for(const until of [NOW+1,NOW-1,0])
test(`${phase} lap blocks publication after expiry as well as before (${until-NOW})`,async()=>{
  const {db,store}=fixture();db.values.set(COMMUNICATIONS_WORKER_LAP,lap(phase,until));
  assert.equal(await store.communicationsLapIdle(),false);
  await assert.rejects(store.acquire('research_release'),/communications_worker_lap_active/);
  assert.equal(db.values.get(ROOT).lease,undefined);
});

const malformed=[x=>delete x.phase,x=>x.phase='completed',x=>delete x.schema_version,
  x=>delete x.lease.generation,x=>x.lease.generation=true,x=>x.lease.generation=0,
  x=>x.lease.owner='other-worker',x=>delete x.lease.until,x=>x.lease.until=false,
  x=>x.lease.until=-1,x=>x.lease.until=NOW-1,x=>x.lease.owner='communications-worker-lap:',
  x=>x.lease.owner='communications-worker-lap:malformed/suffix',x=>delete x.startedAt,
  x=>x.startedAt=true,x=>delete x.renewedAt,x=>x.renewedAt=0.5,x=>delete x.completedAt,
  x=>x.completedAt=null,x=>x.completedAt=Number.MAX_SAFE_INTEGER+1];
for(const [index,change] of malformed.entries())
test(`malformed completion ${index} cannot stand in for confirmed drain`,async()=>{
  const {db,store}=fixture(),record=lap('complete',0);change(record);db.values.set(COMMUNICATIONS_WORKER_LAP,record);
  assert.equal(await store.communicationsLapIdle(),false);
  await assert.rejects(store.acquire('research_release'),/communications_worker_lap_active/);
});

test('only confirmed completion permits publication; default daily owner semantics stay unchanged',async()=>{
  const {db,store}=fixture();db.values.set(COMMUNICATIONS_WORKER_LAP,lap('complete',0));
  assert.equal(await store.communicationsLapIdle(),true);await store.acquire('research_release');
  await store.renew();await store.release();
  db.values.set(COMMUNICATIONS_WORKER_LAP,lap('active',NOW-1));
  await store.acquire();assert.equal(db.values.get(ROOT).lease.owner,'synthetic-publication');
  await assert.rejects(store.transaction(async tx=>{
    await store.publicationFence(tx,(await tx.get(store.control)).data());
  }),/research_release_lease_required/);
});

test('lap introduced after acquire blocks publication renewal without changing lease',async()=>{
  const {db,store}=fixture();await store.acquire('research_release');
  const before=structuredClone(db.values.get(ROOT).lease);db.values.set(COMMUNICATIONS_WORKER_LAP,lap('active',NOW-1));
  await assert.rejects(store.renew(),/communications_worker_lap_active/);
  assert.deepEqual(db.values.get(ROOT).lease,before);await store.release();
  assert.equal(db.values.get(ROOT).lease.expires_at_ms,0);
});

for(const lease of [{owner:'old',generation:1,expires_at_ms:false},{owner:'old',generation:true,expires_at_ms:0},
  {owner:'old',generation:Number.MAX_SAFE_INTEGER,expires_at_ms:0}])
test('malformed canonical release lease cannot authorize a publication claim',async()=>{
  const {db,store}=fixture();db.values.get(ROOT).lease=lease;
  await assert.rejects(store.acquire('research_release'),/firestore_lease_invalid/);
  assert.deepEqual(db.values.get(ROOT).lease,lease);
});

for(const kind of ['screen','team'])
test(`${kind} final control transaction rechecks a lap created after preflight`,async()=>{
  const {db,store}=fixture();await store.acquire('research_release');
  assert.equal(await store.communicationsLapIdle(),true);
  const original=db.runTransaction.bind(db);let injected=false;
  db.runTransaction=(fn,options)=>{
    if(!injected){injected=true;db.values.set(COMMUNICATIONS_WORKER_LAP,lap('active',NOW-1));}
    return original(fn,options);
  };
  if(kind==='screen') {
    const id='a'.repeat(64),current={admission_id:id,generation:'1',bytes:1,
      uri:`gs://blueprint-8c1ca.appspot.com/operations/research/screen-admission/${id}/bundle.json`,
      records:1,direction_sha256:'b'.repeat(64),approval_reference:'synthetic-owner'};
    db.values.get(ROOT).screen_admission={enabled:true,current};
    await assert.rejects(store.screenAdmissionSet(id,{enabled:false,current}),/communications_worker_lap_active/);
    assert.equal(db.values.get(ROOT).screen_admission.enabled,true);
  } else {
    const prior={enabled:true,sha256:'a'.repeat(64),version:1};db.values.get(ROOT).team_universe=prior;
    await assert.rejects(store.teamUniverseSet(prior.sha256,{...prior,enabled:false}),/communications_worker_lap_active/);
    assert.equal(db.values.get(ROOT).team_universe.enabled,true);
  }
});
