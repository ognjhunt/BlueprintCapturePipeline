"""Phase 4: Site-grounded view synthesis.

Pipeline: capture → retrieval index → frame alignment → synthesis

  retrieval_query   — K-NN lookup in site reference index by pose or embedding
  plucker_rays      — 6-channel Plücker ray maps from T_world_camera + intrinsics
  depth_splat       — depth-based forward splatting: warp reference → target viewpoint
  cosmos_inference  — Cosmos-Predict2.5-2B Image2World wrapper (zero-FT)
  synthesize        — top-level orchestrator and CLI
"""
