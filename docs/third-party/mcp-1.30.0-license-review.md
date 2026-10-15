# MCP 1.30.0 license review

Status: pending owner approval. No new owner approval is recorded here.

The main-only dependency gate requires MCP 1.30.0 to clear CVE-2026-59951.
All 12 full-suite shards passed on d0053829db05a988b7be11677a1f010ca5393c45;
source signing then refused the stale exact-version license record.

The reviewed wheel is mcp-1.30.0-py3-none-any.whl, SHA256
`666edb5009503e1047c9d60346a756f94b261f05cc2625f23d41c728ffc484d0`.
Its embedded `mcp-1.30.0.dist-info/licenses/LICENSE` is MIT. Its bytes match
both the [upstream 1.30.0 license](https://github.com/modelcontextprotocol/python-sdk/blob/v1.30.0/LICENSE)
and the [previously approved 1.28.1 license](https://github.com/modelcontextprotocol/python-sdk/blob/v1.28.1/LICENSE).
All three license files are 1,071 bytes, SHA256
`5e13dbbc1d120fc2a03cecde7c91424ae2d7de11b63d58ded2f4431e261ee50d`.
The wheel metadata declares the MIT license classifier. The MIT notice must
remain included with copies or substantial portions of the software.

The [upstream security advisory](https://github.com/modelcontextprotocol/python-sdk/security/advisories/GHSA-84m7-p3x7-pcfv)
identifies 1.30.0 as the patched 1.x release. This review concerns that exact
package and license; it grants no spending, dispatch, capture, or physical-evidence authority.

The proposed policy record stays `approved: false`. The owner decision is
whether to approve MCP 1.30.0 under the unchanged MIT terms for the Blueprint
runtime. Approval would allow source signing and normal deployment to proceed.
