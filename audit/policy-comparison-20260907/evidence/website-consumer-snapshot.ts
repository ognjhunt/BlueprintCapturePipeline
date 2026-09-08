import type {
  TaskEvaluationResultArtifact,
  TaskEvaluationResultEpisode,
  TaskEvaluationResultSiteRecord,
} from "@/lib/taskEvaluationResults";

export type EpisodeFilters = {
  family: string;
  seed: string;
  outcome: "all" | "success" | "failure";
  interpretability: "all" | "interpretable" | "uninterpretable";
};

export type AlignedCanaryCell = {
  key: string;
  cellId: string;
  familyId: string;
  seed: number | null;
  partition: string;
  episodesByCandidate: Record<string, TaskEvaluationResultEpisode | undefined>;
};

export const primaryCanaryDownloadRoles = [
  { key: "summary_csv", label: "Summary CSV", aliases: ["summary_csv"] },
  { key: "episode_csv", label: "Episode CSV", aliases: ["episode_csv"] },
  {
    key: "full_json_report",
    label: "Full JSON",
    aliases: ["full_json_report", "machine_readable_report"],
  },
  { key: "evidence_manifest", label: "Evidence manifest", aliases: ["evidence_manifest"] },
] as const;

function correctionEpisodeKey(candidateId: string, cellId: string, seed: number | undefined) {
  return `${candidateId}\0${cellId}\0${seed ?? "unknown"}`;
}

export function applyPolicyCanaryScoreCorrection(
  result: TaskEvaluationResultSiteRecord,
): TaskEvaluationResultSiteRecord {
  const sidecar = result.score_correction;
  if (
    !sidecar
    || sidecar.correction.source_run_id !== result.publication.run_id
    || sidecar.correction.corrected_result_status !== "completed_unqualified"
    || sidecar.audit.original_publication_preserved !== true
    || sidecar.audit.winner_declared !== false
    || sidecar.correction.score_updates.length !== 20
  ) return result;
  const corrected = structuredClone(result);
  const updates = new Map(sidecar.correction.score_updates.map((update) => [
    correctionEpisodeKey(update.candidate_id, update.cell_id, update.seed),
    update,
  ]));
  const episodes = corrected.publication.result_delivery?.episodes || [];
  for (const episode of episodes) {
    const candidateId = episode.policy_candidate_id || episode.subject_id;
    const update = updates.get(correctionEpisodeKey(
      candidateId,
      episode.variation?.cell_id || "",
      episode.variation?.seed,
    ));
    if (!update) continue;
    const next = update.new_score;
    const measurements = next.measurements || {};
    episode.corrected_score = next;
    episode.score = {
      ...episode.score,
      status: String(next.status || episode.score.status),
      task_succeeded: typeof next.task_succeeded === "boolean"
        ? next.task_succeeded
        : episode.score.task_succeeded,
      progress_score: typeof next.outcome_rank === "number"
        ? Number(next.outcome_rank) / 5
        : episode.score.progress_score,
      destination_error: typeof measurements.final_horizontal_distance_to_destination_m === "number"
        ? measurements.final_horizontal_distance_to_destination_m
        : episode.score.destination_error,
    };
    const failedCriteria = Array.isArray(next.failed_criteria)
      ? next.failed_criteria.map(String)
      : [];
    episode.failure = next.task_succeeded === false ? {
      code: failedCriteria[0] || String(next.outcome || "task_not_complete"),
      phase: "deterministic_score_correction",
      summary: String(
        next.failure_reason_plain_english
          || failedCriteria.map((value) => value.replaceAll("_", " ")).join(", ")
          || "The corrected deterministic score did not satisfy the task contract.",
      ),
    } : null;
  }
  const candidateIds = [...new Set(episodes.map((episode) => (
    episode.policy_candidate_id || episode.subject_id
  )))];
  const correctedCandidates = candidateIds.map((candidateId) => {
    const rows = episodes.filter((episode) => (
      (episode.policy_candidate_id || episode.subject_id) === candidateId
    ));
    const interpretable = rows.filter((episode) => (
      episode.score.policy_outcome_interpretable !== false
    ));
    const successes = interpretable.filter((episode) => episode.score.task_succeeded === true).length;
    const failureCounts: Record<string, number> = {};
    for (const episode of interpretable) {
      for (const criterion of episode.corrected_score?.failed_criteria || []) {
        failureCounts[criterion] = (failureCounts[criterion] || 0) + 1;
      }
    }
    return {
      candidate_id: candidateId,
      episodes_completed: rows.length,
      interpretable_episode_count: interpretable.length,
      success_count: successes,
      success_rate: interpretable.length ? successes / interpretable.length : null,
      failure_counts: failureCounts,
    };
  });
  const projectedCandidates = corrected.publication.policy_canary_result?.candidate_results;
  if (Array.isArray(projectedCandidates)) {
    for (const projected of projectedCandidates) {
      const aggregate = correctedCandidates.find((row) => row.candidate_id === projected.candidate_id);
      if (!aggregate) continue;
      Object.assign(projected, aggregate);
      if (projected.metrics && typeof projected.metrics === "object") {
        Object.assign(projected.metrics, aggregate);
      }
    }
  }
  const deliveredCandidates = corrected.publication.result_delivery?.candidate_results;
  if (Array.isArray(deliveredCandidates)) {
    for (const delivered of deliveredCandidates) {
      const aggregate = correctedCandidates.find((row) => row.candidate_id === delivered.candidate_id);
      if (aggregate) Object.assign(delivered, aggregate);
    }
  }
  if (corrected.publication.result_delivery) {
    corrected.publication.result_delivery.summary.successful_episode_count = episodes.filter(
      (episode) => episode.score.task_succeeded === true,
    ).length;
  }
  if (corrected.publication.policy_canary_result?.counts) {
    corrected.publication.policy_canary_result.counts.completed_learned_policy_rollout_count =
      episodes.length;
  }
  return corrected;
}

export function applyPolicyCanaryEpisodeInterpretation(
  result: TaskEvaluationResultSiteRecord,
): TaskEvaluationResultSiteRecord {
  const sidecar = result.episode_interpretation;
  if (
    !sidecar
    || sidecar.source_binding.record_id !== result.record_id
    || sidecar.source_binding.source_run_id !== result.publication.run_id
    || sidecar.source_binding.source_projection_digest
      !== result.publication.policy_canary_result?.projection_digest
    || sidecar.source_binding.source_delivery_digest
      !== result.publication.result_delivery?.delivery_digest
    || sidecar.source_binding.source_score_correction_sidecar_digest
      !== (result.score_correction?.sidecar_digest || null)
    || sidecar.audit.original_publication_preserved !== true
    || sidecar.audit.deterministic_scores_unchanged !== true
    || sidecar.audit.ranking_or_promotion_effect !== "none"
    || sidecar.episodes.length !== 20
  ) return result;
  const interpreted = structuredClone(result);
  const byEpisode = new Map(sidecar.episodes.map((row) => [
    row.episode_id,
    row.interpretation,
  ]));
  for (const episode of interpreted.publication.result_delivery?.episodes || []) {
    const interpretation = byEpisode.get(episode.episode_id);
    if (interpretation) episode.interpretation = interpretation;
  }
  if (interpreted.publication.policy_canary_result) {
    interpreted.publication.policy_canary_result.episode_interpretation = sidecar.summary;
  }
  return interpreted;
}

const canaryFamilyLabels: Record<string, string> = {
  canonical_anchor: "Baseline anchor",
  placement_approach: "Placement and approach",
  illumination: "Lighting variation",
  camera_sensor: "Camera and sensor variation",
  bounded_physics: "Bounded physics variation",
  admitted_object_material_cousin: "Object and material cousin",
  pairwise_stress: "Combined stress",
  pairwise: "Combined stress",
  held_out_composition: "Held-out composition",
  held_out: "Held-out composition",
};

export function humanCanaryCellLabel(
  row: AlignedCanaryCell,
  index: number,
  rows: AlignedCanaryCell[],
) {
  const base = canaryFamilyLabels[row.familyId]
    || row.familyId.replaceAll("_", " ").replace(/^./, (value) => value.toUpperCase());
  const familyRows = rows.filter((candidate) => candidate.familyId === row.familyId);
  if (familyRows.length < 2) return base;
  const ordinal = rows.slice(0, index + 1)
    .filter((candidate) => candidate.familyId === row.familyId).length;
  return `${base} ${ordinal}`;
}

export const canaryFailureCohorts = [
  "collision",
  "no_motion",
  "action_delivery",
  "contact_loss",
  "task_miss",
  "timeout",
  "camera_sensor",
  "runtime_provider",
  "evidence_gap",
] as const;

export function resolvedCanaryCandidates(result: TaskEvaluationResultSiteRecord) {
  const publication = result.publication;
  if (publication.policy_candidates?.length === 2) return publication.policy_candidates;
  const delivered = publication.result_delivery?.candidate_results || [];
  if (delivered.length === 2) return delivered.map((candidate) => {
    const episode = (publication.result_delivery?.episodes || []).find((row) => (
      row.policy_candidate_id === candidate.candidate_id
    ));
    return {
      candidate_id: candidate.candidate_id,
      display_name: candidate.display_name || candidate.candidate_id,
      checkpoint_digest: candidate.checkpoint_digest
        || episode?.policy_checkpoint_digest
        || "Unavailable — not delivered",
    };
  });
  const projected = publication.policy_canary_result?.candidate_results || [];
  return projected.map((candidate: Record<string, any>) => {
    const metrics = candidate.metrics || {};
    const episode = publication.result_delivery?.episodes.find((row) => (
      row.policy_candidate_id === candidate.candidate_id
    ));
    return {
      candidate_id: candidate.candidate_id,
      display_name: candidate.display_name || metrics.display_name || candidate.candidate_id,
      checkpoint_digest: candidate.checkpoint_digest
        || metrics.checkpoint_digest
        || episode?.policy_checkpoint_digest
        || "Unavailable — not delivered",
    };
  });
}

export function wilson95(successes: number, attempts: number) {
  if (!Number.isInteger(successes) || !Number.isInteger(attempts) || attempts <= 0) return null;
  const boundedSuccesses = Math.min(Math.max(successes, 0), attempts);
  const z = 1.959963984540054;
  const phat = boundedSuccesses / attempts;
  const denominator = 1 + (z * z) / attempts;
  const center = (phat + (z * z) / (2 * attempts)) / denominator;
  const halfWidth = z * Math.sqrt(
    (phat * (1 - phat) + (z * z) / (4 * attempts)) / attempts,
  ) / denominator;
  return {
    lower: Math.max(0, center - halfWidth),
    upper: Math.min(1, center + halfWidth),
  };
}

function episodeMatches(episode: TaskEvaluationResultEpisode, filters: EpisodeFilters) {
  if (filters.family !== "all" && episode.variation?.family_id !== filters.family) return false;
  if (filters.seed !== "all" && String(episode.variation?.seed) !== filters.seed) return false;
  if (filters.outcome === "success" && episode.score.task_succeeded !== true) return false;
  if (filters.outcome === "failure" && episode.score.task_succeeded !== false && !episode.failure) return false;
  if (
    filters.interpretability === "interpretable"
    && episode.score.policy_outcome_interpretable === false
  ) return false;
  if (
    filters.interpretability === "uninterpretable"
    && episode.score.policy_outcome_interpretable !== false
  ) return false;
  return true;
}

export function buildAlignedCanaryCells(
  episodes: TaskEvaluationResultEpisode[],
  candidateIds: string[],
  filters: EpisodeFilters,
) {
  const rows = new Map<string, AlignedCanaryCell>();
  for (const episode of episodes) {
    if (episode.episode_kind !== "learned_candidate" || !episodeMatches(episode, filters)) continue;
    const cellId = episode.variation?.cell_id || "unbound-cell";
    const seed = typeof episode.variation?.seed === "number" ? episode.variation.seed : null;
    const key = `${cellId}\0${seed ?? "unknown"}`;
    const row = rows.get(key) || {
      key,
      cellId,
      familyId: episode.variation?.family_id || "unreported",
      seed,
      partition: episode.variation?.partition || "unreported",
      episodesByCandidate: {},
    };
    const candidateId = episode.policy_candidate_id || episode.subject_id;
    if (candidateIds.includes(candidateId)) row.episodesByCandidate[candidateId] = episode;
    rows.set(key, row);
  }
  const quickCellIndex = (cellId: string) => {
    const match = cellId.match(/(?:^|\.)quick10\.(\d{1,3})(?:\.|$)/);
    return match ? Number(match[1]) : Number.MAX_SAFE_INTEGER;
  };
  return [...rows.values()].sort((left, right) => (
    quickCellIndex(left.cellId) - quickCellIndex(right.cellId)
    || left.cellId.localeCompare(right.cellId)
    || (left.seed ?? Number.MAX_SAFE_INTEGER) - (right.seed ?? Number.MAX_SAFE_INTEGER)
  ));
}

function failureCohort(episode: TaskEvaluationResultEpisode): typeof canaryFailureCohorts[number] | null {
  const material = [
    episode.failure?.code,
    episode.failure?.phase,
    episode.failure?.summary,
    episode.action_delivery?.harness_failure_code,
    episode.evidence?.typed_media_gap?.code,
  ].filter(Boolean).join(" ").toLowerCase();
  if (episode.score.collision === true || material.includes("collision")) return "collision";
  if (material.includes("no_motion") || material.includes("no motion") || (
    episode.action_delivery?.actions_reached_robot === true
    && episode.action_delivery.arm_moved === false
  )) return "no_motion";
  if (episode.action_delivery?.actions_reached_robot === false || material.includes("action_delivery")) return "action_delivery";
  if (material.includes("contact_loss") || material.includes("contact loss")) return "contact_loss";
  if (material.includes("task_miss") || material.includes("task miss")) return "task_miss";
  if (material.includes("timeout")) return "timeout";
  if (material.includes("camera") || material.includes("sensor")) return "camera_sensor";
  if (material.includes("runtime") || material.includes("provider")) return "runtime_provider";
  if (episode.evidence?.typed_media_gap || material.includes("evidence") || material.includes("media")) return "evidence_gap";
  return episode.failure ? "task_miss" : null;
}

export function buildFailureAnalysis(episodes: TaskEvaluationResultEpisode[]) {
  const cohorts = new Map<typeof canaryFailureCohorts[number], string[]>();
  for (const cohort of canaryFailureCohorts) cohorts.set(cohort, []);
  for (const episode of episodes) {
    const cohort = failureCohort(episode);
    if (cohort) cohorts.get(cohort)?.push(episode.episode_id);
  }
  return canaryFailureCohorts.map((cohort) => ({
    cohort,
    count: cohorts.get(cohort)?.length || 0,
    representativeEpisodeIds: (cohorts.get(cohort) || []).slice(0, 3),
  }));
}

function episodeArtifacts(episode: TaskEvaluationResultEpisode) {
  return [
    episode.artifacts?.receipt,
    episode.artifacts?.frame_manifest,
    ...Object.values(episode.artifacts?.videos || {}),
    episode.evidence?.lossless_policy_inputs,
    episode.evidence?.frame_manifest,
    ...Object.values(episode.evidence?.videos || {}),
    episode.evidence?.episode_json,
    episode.evidence?.indexed_mcap_rosbag,
    episode.action_delivery?.returned_action_sequence,
    episode.action_delivery?.delivery_readback,
    episode.traces?.state,
    episode.traces?.contact_force,
    episode.traces?.task_object_trajectory,
  ].filter((artifact): artifact is TaskEvaluationResultArtifact => Boolean(artifact));
}

export function normalizedArtifact(value: unknown): TaskEvaluationResultArtifact | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  const artifactId = typeof record.artifact_id === "string"
    ? record.artifact_id.trim()
    : "";
  if (!artifactId) return null;
  const digest = typeof record.sha256 === "string"
    ? record.sha256
    : typeof record.digest === "string"
      ? record.digest
      : "";
  return {
    ...record,
    artifact_id: artifactId,
    role: typeof record.role === "string" && record.role.trim()
      ? record.role
      : "unclassified_artifact",
    relative_path: typeof record.relative_path === "string" && record.relative_path.trim()
      ? record.relative_path
      : artifactId,
    sha256: digest,
    size_bytes: typeof record.size_bytes === "number" ? record.size_bytes : 0,
    content_type: typeof record.content_type === "string" && record.content_type.trim()
      ? record.content_type
      : typeof record.media_type === "string" && record.media_type.trim()
        ? record.media_type
        : "application/octet-stream",
  } as TaskEvaluationResultArtifact;
}

function artifactQuality(value: unknown) {
  if (!value || typeof value !== "object") return 0;
  const record = value as Record<string, unknown>;
  return [
    record.role,
    record.relative_path,
    record.sha256 || record.digest,
    record.size_bytes,
    record.content_type || record.media_type,
  ].filter((field) => field !== null && field !== undefined && field !== "").length;
}

export function buildCanaryArtifactInventory(result: TaskEvaluationResultSiteRecord) {
  const publication = result.publication;
  const delivery = publication.result_delivery;
  const canaryResult = publication.policy_canary_result || {};
  const reproducibility = canaryResult.reproducibility || {};
  const artifacts = [
    ...(delivery?.artifacts || []),
    ...(delivery?.episodes || []).flatMap(episodeArtifacts),
    canaryResult.report?.machine_readable_report,
    canaryResult.report?.evidence_manifest,
    canaryResult.report?.controls_csv,
    ...(canaryResult.controls || delivery?.controls || []).flatMap((control: import("./policyCanaryControls").PolicyCanaryControl) => [control.receipt, control.cell_receipt, ...Object.values(control.videos), ...control.artifacts]),
    canaryResult.closure?.billing,
    canaryResult.closure?.teardown,
    canaryResult.closure?.provider_zero,
    reproducibility.evidence_manifest,
    reproducibility.billing_receipt,
    reproducibility.teardown_receipt,
    reproducibility.provider_zero_receipt,
    publication.notification_delivery?.receipt,
  ].filter((artifact): artifact is TaskEvaluationResultArtifact => Boolean(artifact));
  const unique = new Map<string, { artifact: TaskEvaluationResultArtifact; quality: number }>();
  for (const value of artifacts) {
    const artifact = normalizedArtifact(value);
    if (!artifact) continue;
    const quality = artifactQuality(value);
    const current = unique.get(artifact.artifact_id);
    if (!current || quality > current.quality) {
      unique.set(artifact.artifact_id, { artifact, quality });
    }
  }
  return [...unique.values()].map(({ artifact }) => artifact).sort((left, right) => (
    left.role.localeCompare(right.role) || left.relative_path.localeCompare(right.relative_path)
  ));
}

export function primaryCanaryDownloads(result: TaskEvaluationResultSiteRecord) {
  const inventory = buildCanaryArtifactInventory(result);
  return primaryCanaryDownloadRoles.map((download) => ({
    key: download.key,
    label: download.label,
    artifact: inventory.find((artifact) => download.aliases.some(
      (alias) => alias === artifact.role,
    )) || null,
  }));
}

export function availableCanaryFilters(episodes: TaskEvaluationResultEpisode[]) {
  return {
    families: [...new Set(episodes.map((episode) => episode.variation?.family_id).filter(Boolean) as string[])].sort(),
    seeds: [...new Set(episodes.map((episode) => episode.variation?.seed).filter((seed): seed is number => typeof seed === "number"))].sort((a, b) => a - b),
  };
}

// --- Answer-first comparison helpers ---------------------------------------
// The metrics a robot team reads first: each policy's success rate (k/N) with a
// Wilson interval, and a paired verdict that says whether the observed gap is
// distinguishable at this sample size. All computed from delivered episodes and
// candidate results — no figure is invented, and the deterministic scores are
// never changed here.

export type CanaryCandidateSummary = {
  candidate_id: string;
  display_name: string;
  success_count: number;
  interpretable_count: number;
  success_rate: number | null;
  wilson: { lower: number; upper: number } | null;
};

function numeric(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function canaryCandidateSummaries(
  result: TaskEvaluationResultSiteRecord,
): CanaryCandidateSummary[] {
  const publication = result.publication;
  const delivery = publication.result_delivery;
  const canary = publication.policy_canary_result || {};
  const candidateResults: Array<Record<string, any>> = canary.candidate_results?.length
    ? canary.candidate_results
    : delivery?.candidate_results || [];
  const byId = new Map(candidateResults.map((row) => [row.candidate_id, row]));
  const episodes = delivery?.episodes || [];
  return resolvedCanaryCandidates(result).map((candidate) => {
    const row = byId.get(candidate.candidate_id);
    const metrics = (row?.metrics as Record<string, unknown>) || {};
    let successCount = numeric(row?.success_count ?? metrics.success_count);
    let interpretable = numeric(
      row?.interpretable_episode_count ?? metrics.interpretable_episode_count,
    );
    if (successCount === null || interpretable === null) {
      const rows = episodes.filter((episode) => (
        episode.episode_kind === "learned_candidate"
        && (episode.policy_candidate_id || episode.subject_id) === candidate.candidate_id
        && episode.score.policy_outcome_interpretable !== false
      ));
      interpretable = rows.length;
      successCount = rows.filter((episode) => episode.score.task_succeeded === true).length;
    }
    return {
      candidate_id: candidate.candidate_id,
      display_name: candidate.display_name,
      success_count: successCount,
      interpretable_count: interpretable,
      success_rate: interpretable > 0 ? successCount / interpretable : null,
      wilson: wilson95(successCount, interpretable),
    };
  });
}

function choose(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  let result = 1;
  for (let i = 0; i < k; i += 1) result = (result * (n - i)) / (i + 1);
  return result;
}

// Two-sided exact sign test on discordant pairs — the correct paired comparison
// when both policies run the same matched cells (McNemar's exact form).
function twoSidedSignTestP(a: number, b: number): number | null {
  const n = a + b;
  if (n <= 0) return null;
  const k = Math.min(a, b);
  let tail = 0;
  for (let i = 0; i <= k; i += 1) tail += choose(n, i);
  return Math.min(1, 2 * tail * Math.pow(0.5, n));
}

export type PairedCanaryComparison = {
  candidates: [CanaryCandidateSummary, CanaryCandidateSummary];
  leader: CanaryCandidateSummary | null;
  deltaPoints: number | null;
  comparablePairs: number;
  bothSucceeded: number;
  bothFailed: number;
  leaderOnlyWins: number;
  laggardOnlyWins: number;
  discordantPairs: number;
  pValue: number | null;
  distinguishable: boolean;
  headline: string;
  verdict: string;
};

export function pairedCanaryComparison(
  result: TaskEvaluationResultSiteRecord,
): PairedCanaryComparison | null {
  const summaries = canaryCandidateSummaries(result);
  if (summaries.length !== 2) return null;
  const [candidateA, candidateB] = summaries;
  const episodes = result.publication.result_delivery?.episodes || [];
  const rows = buildAlignedCanaryCells(
    episodes,
    [candidateA.candidate_id, candidateB.candidate_id],
    { family: "all", seed: "all", outcome: "all", interpretability: "all" },
  );
  let comparablePairs = 0;
  let aOnlyWins = 0;
  let bOnlyWins = 0;
  let bothSucceeded = 0;
  let bothFailed = 0;
  for (const row of rows) {
    const a = row.episodesByCandidate[candidateA.candidate_id];
    const b = row.episodesByCandidate[candidateB.candidate_id];
    if (!a || !b) continue;
    if (
      a.score.policy_outcome_interpretable === false
      || b.score.policy_outcome_interpretable === false
    ) continue;
    if (typeof a.score.task_succeeded !== "boolean" || typeof b.score.task_succeeded !== "boolean") {
      continue;
    }
    comparablePairs += 1;
    const aWin = a.score.task_succeeded;
    const bWin = b.score.task_succeeded;
    if (aWin && bWin) bothSucceeded += 1;
    else if (!aWin && !bWin) bothFailed += 1;
    else if (aWin) aOnlyWins += 1;
    else bOnlyWins += 1;
  }
  const discordantPairs = aOnlyWins + bOnlyWins;
  const pValue = twoSidedSignTestP(aOnlyWins, bOnlyWins);
  const distinguishable = pValue !== null && pValue < 0.05;

  let leader: CanaryCandidateSummary | null = null;
  let deltaPoints: number | null = null;
  if (candidateA.success_rate !== null && candidateB.success_rate !== null) {
    if (candidateA.success_rate === candidateB.success_rate) {
      deltaPoints = 0;
    } else {
      leader = candidateA.success_rate > candidateB.success_rate ? candidateA : candidateB;
      deltaPoints = Math.round(
        Math.abs(candidateA.success_rate - candidateB.success_rate) * 100,
      );
    }
  }
  const leaderWins = leader?.candidate_id === candidateA.candidate_id ? aOnlyWins : bOnlyWins;
  const laggardWins = leader?.candidate_id === candidateA.candidate_id ? bOnlyWins : aOnlyWins;

  const pText = pValue === null
    ? null
    : pValue < 0.001
      ? "p < 0.001"
      : `p ≈ ${pValue.toFixed(2)}`;

  let headline: string;
  let verdict: string;
  if (leader && deltaPoints) {
    headline = `${leader.display_name} led by ${deltaPoints} pp`;
    if (comparablePairs === 0) {
      verdict = "No matched, scorable cells were available to test the gap.";
    } else if (distinguishable) {
      verdict = `The gap is statistically distinguishable on matched cells (exact sign test ${pText}).`;
    } else {
      verdict = `The gap is not statistically distinguishable at this sample size — a diagnostic signal, not a declared winner (exact sign test ${pText}).`;
    }
  } else if (deltaPoints === 0) {
    headline = "Both policies scored the same success rate";
    verdict = "No separation on this scene at this sample size.";
  } else {
    headline = "Comparative success rate not delivered for both policies";
    verdict = "At least one policy did not deliver a scorable success rate.";
  }

  return {
    candidates: [candidateA, candidateB],
    leader,
    deltaPoints,
    comparablePairs,
    bothSucceeded,
    bothFailed,
    leaderOnlyWins: leaderWins,
    laggardOnlyWins: laggardWins,
    discordantPairs,
    pValue,
    distinguishable,
    headline,
    verdict,
  };
}

export function formatCanaryPercent(rate: number | null): string {
  return rate === null ? "Not scored" : `${Math.round(rate * 100)}%`;
}
