// Read the Website owner's committed fix without touching their active worktree.
const fs = require('node:fs');
const cp = require('node:child_process');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const esbuild = require('/Users/nijelhunt_1/workspace/Blueprint-WebApp/node_modules/esbuild');
const commit = '29bf82389a751c94e401959c47105a442e7fc9a9';
const source = cp.execFileSync('git', ['-C', '/Users/nijelhunt_1/workspace/Blueprint-WebApp',
  'show', `${commit}:client/src/lib/policyCanaryResultPortal.ts`], {encoding:'utf8'});
const output = esbuild.transformSync(source, {loader:'ts',format:'cjs'}).code;
const compiled = {exports:{}};
new Function('module','exports',output)(compiled, compiled.exports);
const ids = ['pi05_droid','groot_n17_droid'];
const episodes = Array.from({length:10},(_,cell)=>ids.map((candidate,index)=>{
  const valid=index===0?cell<6:cell<2||cell>=6;
  const win=index===0?cell<2:cell>=6;
  return {episode_id:`${candidate}-${cell}`,episode_kind:'learned_candidate',subject_id:candidate,
    policy_candidate_id:candidate,variation:{cell_id:`cell-${cell}`,seed:100+cell,family_id:'canonical_anchor',partition:'canonical'},
    score:{status:valid?'scored':'undetermined',task_succeeded:valid?win:null,policy_outcome_interpretable:valid,grader_authority:'deterministic_simulator_state'},
    evidence:{complete:valid}};
})).flat();
const packet={publication:{policy_candidates:ids.map(id=>({candidate_id:id,display_name:id})),result_delivery:{episodes}}};
const compare=compiled.exports.pairedCanaryComparison;
const paired=compare(packet);
assert.equal(paired.leader.candidate_id, ids[0]);
assert.equal(paired.deltaPoints,100);
assert.equal(paired.comparablePairs,2);
assert.equal(paired.pValue,.5);
const reversed=structuredClone(packet);
reversed.publication.result_delivery.episodes.reverse();
reversed.publication.policy_candidates.reverse();
assert.equal(compare(reversed).leader.candidate_id,ids[0]);
assert.equal(compare(reversed).deltaPoints,100);
const empty=structuredClone(packet);
for(const row of empty.publication.result_delivery.episodes) {
  if(row.policy_candidate_id===ids[1] && row.variation.seed<102) row.score.policy_outcome_interpretable=false;
}
assert.equal(compare(empty).comparablePairs,0);
assert.equal(compare(empty).leader,null);
process.stdout.write(JSON.stringify({fixture_only:true,website_commit:commit,
  source_sha256:crypto.createHash('sha256').update(source).digest('hex'),
  verified:['asymmetric_missingness','candidate_and_episode_order','no_shared_pairs'],comparison:paired},null,2)+'\n');
