# Protocol Amendment 29: policy query 5 result and WAM5

Status: frozen after policy query 5 and before WAM5 provider execution

Date: 2026-08-01

## Policy-query-5 result

The first query-5 live invocation at immutable runtime SHA
`4f6ed4f9654298b474f89a4e1370eb9ca8623d05` stopped before provider
mutation because the then-current OpenPI refresh gate still required global
provider zero. An unrelated Vast GPU was live. No campaign reservation,
watchdog, provider allocation, authorization consumption, or spend occurred.
That preserved failure motivated the generic, prospectively tested two-GPU
admission correction in Amendment 28; it did not change the scientific input.

The scientifically identical query-5 request then completed at immutable pushed
runtime SHA `ede38013d6cb2a5453ed39ba39c607a7f497a639` on owned Vast
instance `46507832`. The same frozen `pi05_droid` checkpoint was queried
using only WAM4's three generated final camera frames and registered
commanded-prefix state. No future physical RGB, future recorded state, physical
outcome, or WAM execution entered the policy job.

The provider archive SHA-256 is
`9ab4d88e6b8b3b65afba65ca760fbc84167fe75f211f5c2599c0a368daeca1b2`.
The complete native 15x8 action file SHA-256 is
`56619489601e197dea4b06a87f26f3e5d7ed59f219e056ff7e222a91fc3ba126`;
its deterministic native-action content SHA-256 is
`14a4f649d514561b083cd1ff8577caef99baa362aa7322ef355095500249855d`.
The deterministic policy request SHA-256 is
`9925ac01333c881cad079b12386f590704fe9f3650cc18e2560a7edaf0347e67`,
and policy identity SHA-256 remained
`ef2133d7cde82ef08bd9d0cabc7091cab9c4d80779e544c19831c23ff9f15fb8`.

The single-use authorization was atomically consumed before provider mutation;
its nondisclosing consumption-record SHA-256 is
`8d22ead4347da92b513467f990519f9e93050146fe57f5de6b8d46d35ca29d03`.
The run charged 162 GPU-seconds and USD `0.033750`. The authoritative
production ledger now commits 36,723 GPU-seconds and USD `10.727231`, with no
open reservation and USD `9.272769` remaining under its conservative USD 20
internal GPU cap. Together with the unchanged evaluator/API ledger of USD
`8.418512`, cumulative GPU plus evaluator spend is USD `19.145743`.
The watchdog destroyed the owned instance, and the all-provider inventory
proved zero after closure with a campaign-wide maximum of two GPUs.

Preserved evidence includes:

- policy receipt file SHA-256
  `4e269bd23307fb0e17e5f93d84a3563c7726b59715dd908cb6ec17609ed0be25`;
- policy receipt manifest SHA-256
  `551cd1ed37d9296f185f264b3b3660edec315ced4fa67c6d2b498b298f9d8806`;
- output validation SHA-256
  `606c078cf22b2bd366332959b719edbfd838b1e39817878a450e1bd0e1fa6df8`;
- monitor SHA-256
  `c831099448a192cc0aca4a226cd7d23e1c961f7f49d71c180d8d80dcf25693fd`;
- production budget ledger SHA-256
  `56398c99daad4fb70b05b07cb15bbccdac16bcf4179d6ac1790ff2d0709ce2aa`;
- independently extracted output receipt SHA-256
  `9e3f50362e4568892322295cf1639eca9cb1e0e78ff6d6a8c0a77234756ba314`;
  and
- provider-zero receipt SHA-256
  `e496f9aef5ad796c6c9d7006a608a0bdf3bd4f49ef4276a1884574e16567397e`.

This completes interaction five. It does not complete the 12-interaction
episode, causal qualification, ranking, blind confirmation, captured-site
transfer, or economics.

## Frozen WAM5 input

WAM5 begins interaction six of the unchanged 12-interaction complete horizon.
It conditions on policy query 5's complete native action through the released
Ctrl-World joint-velocity adapter. Its three view histories and commanded
Cartesian state history each contain 29 rows: the frozen 24-frame initial
history followed by the final WAM0 through WAM4 feedback states. The 28-row
state-history prefix is byte-identical to the frozen WAM4 input, and its
appended commanded state is exactly transition 4's registered next state.

No future physical RGB, future recorded state, outcome label, policy identity,
or recorded action trace enters the WAM request. Selection was not based on
numeric action values or WAM output; WAM5 output did not exist at freeze time.

The immutable bindings are:

- released-adapter conditioning SHA-256
  `456d0434e572e6dc3eb7cc570ab894edf6def58e45ffd125b5511c99e7a88f45`;
- WAM5 request SHA-256
  `5747e6dc6975c405b9c92ef2f275dbb5dab6072253327dbfbfef88653217725d`;
- request-manifest file SHA-256
  `445fc1009c68acf02639325c8a1ff94937d4bd6660f0acc59f967f1fcdcfcc82`;
- transition-evidence manifest SHA-256
  `4dcd5cf3b8ff0b5bf2ff872ef53d114ee64737fc8c0a6e1c05cd54a694344e09`;
- transition-evidence file SHA-256
  `b4490c444f7d4f960ea1722b1609e27f1613fb3d8f868ca65ba1a5c75e85f22f`;
- transition-freeze receipt manifest SHA-256
  `0aba355b83350940b5aae28726f7b071b2875f62c054e9caf648cdd6bbfbc474`;
- transition-freeze receipt file SHA-256
  `89526a77027deb37a3bb42709acd179508da348a4d1285c3f9575309bb52ec4f`;
- provider-bundle SHA-256
  `e7942b014e9acd11930236b8d7a98200ec13892b1723269c7b24197ac6c918f0`;
  and
- provider-bundle receipt file SHA-256
  `3d2476cc285f25613378b3136b621ed2de19fda30bd086993ea81cb1f9d1bfa5`.

Twenty-two focused transition/request tests pass. Ninety focused successor
admission, allocator, and cumulative-campaign-budget tests pass at immutable
pushed experiment runtime SHA
`17ed9db13d3c98e45d09faa2bb946ec48e716e59`. The admission contract permits
at most two globally live GPUs, proves one existing resource admits a second
and two existing resources block a third, while WAM5 itself remains limited to
exactly one allocation and one GPU.

## Paid execution boundary

WAM5 may execute only from a new immutable pushed experiment SHA through the
canonical paid-resource allocator after fresh credential, ledger, global
inventory, transport, provider preflight, and dry-run checks. It retains the
USD 5 allocation cap, USD 3 target, USD 2.05/hour offer ceiling, 4,800-second
hard TTL, real production-campaign reservation, independent watchdog, teardown,
and provider-zero requirements.

## Decision boundary

If WAM5 completes and its immediate reliability gate passes, Blueprint may
construct only the registered generated observation and same-policy query 6.
Gemini 3.6 Flash and GPT-5.6 Luna remain forbidden until the complete
12-interaction episode and causal-control matrix both pass.
