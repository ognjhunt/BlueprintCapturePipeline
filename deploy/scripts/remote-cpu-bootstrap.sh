#!/bin/bash
# Sourced by deploy.sh after the unchanged clean-main/full-lane release gates.
# ADP-009D/day 28: Terraform owns two isolated projects in the canonical state.

verify_remote_cpu_image_revision() {
    local image="$1" source_commit="$2"
    [[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || return 2
    command -v crane >/dev/null || {
        printf '%s\n' 'crane is required to verify the immutable worker image revision.' >&2
        return 2
    }
    if ! (set -o pipefail; crane config --platform linux/amd64 "$image" |
        jq -e --arg source "$source_commit" \
            '.config.Labels["org.opencontainers.image.revision"] == $source' >/dev/null); then
        printf '%s\n' 'Worker image revision is missing or differs from promoted source.' >&2
        return 2
    fi
}

apply_remote_cpu_bootstrap() {
    [[ "${REMOTE_CPU_WORKERS_ENABLED}" == "true" ]] || {
        log_error "The reviewed worker bootstrap requires remote CPU workers enabled."
        return 2
    }
    local account
    account="$(gcloud auth list --filter=status:ACTIVE --format='value(account)')"
    [[ "$(printf %s "$account" | tr "[:upper:]" "[:lower:]")" == "ohstnhunt@gmail.com" ]] || {
        log_error "The isolated project must be created with the founder's user credentials."
        return 2
    }
    local tagged_image="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"
    local resolved_image
    resolved_image="$(resolve_image_digest_uri "$tagged_image")"
    require_resolved_image_matches "exact-release pipeline image" "$IMAGE_DIGEST_URI" "$resolved_image"
    verify_remote_cpu_image_revision "$resolved_image" "$GIT_SHA" || return 2
    export TF_VAR_project_id="$PROJECT_ID"
    export TF_VAR_primary_region="$PRIMARY_REGION"
    export TF_VAR_deployment_scope=remote_cpu
    export TF_VAR_remote_cpu_workers_enabled=true
    export TF_VAR_remote_cpu_worker_object_prefix="$REMOTE_CPU_WORKER_OBJECT_PREFIX"
    export TF_VAR_docker_image="$resolved_image"
    local billing_account
    billing_account="$(gcloud billing projects describe "$PROJECT_ID" --format='value(billingAccountName)')"
    export TF_VAR_billing_account_id="${billing_account#billingAccounts/}"
    [[ "$TERRAFORM_STATE_BUCKET" == "blueprint-8c1ca-terraform-state" && "$TERRAFORM_STATE_PREFIX" == "capture-pipeline" ]] || {
        log_error "Worker bootstrap requires the canonical capture-pipeline backend."
        return 2
    }
    # Inactive image/secret settings are absent, never fabricated or borrowed.
    export TF_VAR_privacy_sam3_image="" TF_VAR_privacy_vip_image=""
    export TF_VAR_privacy_deepprivacy2_image="" TF_VAR_video_to_world_image=""
    export TF_VAR_worldlabs_api_key_secret_name="" TF_VAR_privacy_runner_token_secret_name=""
    export TF_VAR_pipeline_sync_token_secret_name="" TF_VAR_pipeline_sync_webapp_url=""
    validate_terraform_state_backend
    cd "$TERRAFORM_DIR"
    terraform init -input=false -reconfigure \
        "-backend-config=bucket=${TERRAFORM_STATE_BUCKET}" \
        "-backend-config=prefix=${TERRAFORM_STATE_PREFIX}" \
        "-backend-config=kms_encryption_key=${TERRAFORM_STATE_KMS_KEY}"
    local validator="$PROJECT_ROOT/scripts/remote_cpu_deployment_scope.py"
    local evidence_dir
    evidence_dir="$(dirname "$TOPOLOGY_EVIDENCE_PATH")"
    mkdir -p "$evidence_dir"
    # Saved plans can contain sensitive configuration. Restrict every artifact.
    umask 077
    terraform show -json > "$evidence_dir/remote-cpu-preapply-state.json"
    python3 "$validator" state "$evidence_dir/remote-cpu-preapply-state.json"
    local -a targets=()
    local target
    while IFS= read -r target; do targets+=("$target"); done < <(python3 "$validator" targets)
    [[ ${#targets[@]} -gt 0 ]] || return 2
    terraform plan -input=false "${targets[@]}" -out=remote-cpu.tfplan
    terraform show -json remote-cpu.tfplan > "$evidence_dir/remote-cpu-plan.json"
    python3 "$validator" plan "$evidence_dir/remote-cpu-plan.json"
    if [[ "$DRY_RUN" == "true" ]]; then
        log_success "Worker-only plan validated; no resources were applied."
        return 0
    fi
    confirm "Apply the reviewed isolated worker scope?" || return 2
    terraform apply -input=false remote-cpu.tfplan
    rm -f remote-cpu.tfplan
    # Both projects have no inherited parents. Runtime is founder + Run agent;
    # dispatch/transport is founder only, with no runtime API or service agent.
    local isolated_project project_number
    for isolated_project in blueprint-remote-cpu-8c1ca blueprint-cpu-dispatch-8c1ca; do
        gcloud projects describe "$isolated_project" --format=json > "$evidence_dir/$isolated_project.json"
        jq -e '(.parent // null) == null' "$evidence_dir/$isolated_project.json" >/dev/null
        project_number="$(jq -r .projectNumber "$evidence_dir/$isolated_project.json")"
        gcloud projects get-iam-policy "$isolated_project" --format=json > "$evidence_dir/$isolated_project-iam.json"
        local -a iam_options=(--project-number "$project_number")
        [[ "$isolated_project" != "blueprint-cpu-dispatch-8c1ca" ]] || iam_options+=(--dispatch-project)
        python3 "$validator" project-iam "$evidence_dir/$isolated_project-iam.json" "${iam_options[@]}"
    done
    # The first bootstrap identity must remain keyless and unused.
    gcloud iam service-accounts keys list \
        --iam-account=remote-cpu-dispatcher@blueprint-remote-cpu-8c1ca.iam.gserviceaccount.com \
        --managed-by=user --format=json > "$evidence_dir/quarantined-dispatcher-user-keys.json"
    jq -e 'length == 0' "$evidence_dir/quarantined-dispatcher-user-keys.json" >/dev/null
    local drift_exit=0
    terraform plan -input=false "${targets[@]}" -detailed-exitcode -out=remote-cpu-drift.tfplan || drift_exit=$?
    [[ $drift_exit -eq 0 ]] || {
        log_error "The worker scope has drift or its refresh failed."
        return 2
    }
    terraform show -json remote-cpu-drift.tfplan > "$evidence_dir/remote-cpu-drift.json"
    python3 "$validator" plan "$evidence_dir/remote-cpu-drift.json"
    terraform output -json remote_cpu_bootstrap_scope > "$evidence_dir/remote-cpu-scope.json"
    jq -n --arg git_sha "$GIT_SHA" --arg image "$resolved_image" \
        --arg full_lane "$FULL_TEST_LANE_EVIDENCE_URI" \
        '{schema_version:"remote_cpu_topology_evidence.v1",scope:"remote_cpu",
          git_sha:$git_sha,image:$image,full_test_lane:$full_lane,
          project:"blueprint-remote-cpu-8c1ca",dispatch_project:"blueprint-cpu-dispatch-8c1ca",
          transport_bucket:"blueprint-cpu-dispatch-8c1ca-transport",provider_refresh_zero_drift:true,
          runtime_agent_trust:"May impersonate Worker to read inputs; cannot mutate transport, impersonate Dispatcher or execute jobs.",
          claim_boundary:"Isolated worker infrastructure only. Real IAM fence checks, credentials, paid preflight and shadow parity remain separate gates."}' \
        > "$TOPOLOGY_EVIDENCE_PATH"
    rm -f remote-cpu-drift.tfplan
    log_success "Isolated worker topology deployed with a zero-change scoped refresh."
}
