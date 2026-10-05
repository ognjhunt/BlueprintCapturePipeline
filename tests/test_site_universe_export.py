"""Backlog export: the producer -> runtime contract, refusals, lead capability and CLI (hermetic fixtures)."""
import gzip
import hashlib
import json
import shutil

import pytest

from tests.site_universe_fixture import seeded_cache
from tools.daily_research import site_universe as runtime
from tools.site_universe import build, export, rank
from tools.site_universe.cli import main

APPROVAL = "owner-synthetic-ranking-review"


@pytest.fixture(scope="module")
def ranked(tmp_path_factory):
    root = tmp_path_factory.mktemp("export")
    snapshot = build.build(["TX"], out_dir=root / "snap", cache=seeded_cache(root / "raw"))
    return snapshot, rank.write_ranking(snapshot.out_dir, root / "rank")


def exported(ranked, out, **options):
    snapshot, run = ranked
    return export.export(run.out_dir, snapshot.out_dir, out, approval_reference=APPROVAL, **options)


def ranked_rows(run):
    rows = [json.loads(line) for line in gzip.decompress(run.ranked_path.read_bytes()).decode().splitlines()]
    return [row for row in rows if row["status"] == "ranked"]


def pin_for(raw, manifest, slice_size=5):
    sha = hashlib.sha256(raw).hexdigest()
    return {"enabled": True, "uri": runtime.object_uri(sha), "generation": "1001", "sha256": sha, "bytes": len(raw),
            "snapshot_id": manifest["snapshot_id"], "rank_config_sha256": manifest["rank_config_sha256"],
            "slice_size": slice_size, "reoffer_after_days": 90, "approval_reference": "owner-synthetic-pin"}


def test_the_fixture_snapshot_ranks_exports_loads_and_selects_end_to_end(ranked, tmp_path):
    snapshot, run = ranked
    summary = exported(ranked, tmp_path / "out")
    raw = (tmp_path / "out" / runtime.OBJECT_NAME).read_bytes()
    assert summary["sha256"] == hashlib.sha256(raw).hexdigest() and summary["bytes"] == len(raw)
    assert summary["uri"] == runtime.object_uri(summary["sha256"])
    pin = pin_for(raw, json.loads(gzip.decompress(raw))["manifest"])
    loaded = runtime.load_export(raw, runtime.pin(pin, 2700))  # The runtime loader accepts it, bound to its pin.
    manifest, rows, source = loaded["manifest"], loaded["rows"], ranked_rows(run)
    assert manifest["snapshot_id"] == snapshot.snapshot_id == summary["snapshot_id"]
    assert manifest["counts"] == {"sites": run.manifest["counts"]["sites_in_snapshot"],
                                  "ranked": run.manifest["counts"]["sites_ranked"],
                                  "excluded": run.manifest["counts"]["sites_excluded"], "rows": len(rows)}
    assert sum(manifest["lead_capability_counts"].values()) == len(source)
    assert manifest["license_union"] and set(manifest["license_union"]) <= runtime.LICENSES
    assert manifest["distribution"] == "internal_only" and manifest["approval_reference"] == APPROVAL
    assert manifest["rank_config_sha256"] == run.manifest["rank_config"]["sha256"]
    assert manifest["selection_policy"]["seed_capabilities"][0] == "fixed_arm_machine_tending"
    assert manifest["previous_snapshot_id"] is None and manifest["new_sites"] is None
    # The fixture ranking fits inside the default top 3000: every ranked row, in rank order.
    assert [row["site_id"] for row in rows] == [row["site_id"] for row in source]
    by_id = {row["site_id"]: row for row in source}
    for row in rows:
        assert row["lead_capability"] == rank.lead_capability(by_id[row["site_id"]])
        assert row["rank"] == by_id[row["site_id"]]["rank"] and row["score"] == by_id[row["site_id"]]["score"]
        assert row["name"] not in row["aliases"] and len(row["fit"]) <= 240
    sites, selection = runtime.select(loaded, history=[], crm_values=[], run_date="2026-10-06", slice_size=5,
                                      reoffer_after_days=90)
    assert 0 < len(sites) == selection["offered"] <= 5 and selection["eligible"] == len(rows)
    assert [site["rank"] for site in sites] == sorted(site["rank"] for site in sites)
    exported(ranked, tmp_path / "again")
    assert (tmp_path / "again" / runtime.OBJECT_NAME).read_bytes() == raw  # Byte-identical for the same inputs.


def test_top_and_per_capability_select_rows_in_global_rank_order(ranked, tmp_path):
    _, run = ranked
    exported(ranked, tmp_path / "out", top=2, per_capability=1)
    rows = runtime.load_export((tmp_path / "out" / runtime.OBJECT_NAME).read_bytes())["rows"]
    expected, seen = [], set()
    for row in ranked_rows(run):
        lead = rank.lead_capability(row)
        if row["rank"] <= 2 or lead not in seen:
            expected.append(row["site_id"])
        seen.add(lead)
    assert [row["site_id"] for row in rows] == expected and len(expected) < len(ranked_rows(run))
    with pytest.raises(export.ExportError, match="limit is 1"):
        exported(ranked, tmp_path / "small", top=2, per_capability=1, max_rows=1)


def test_export_refuses_an_out_dir_inside_the_repository(ranked):
    inside = rank.RANK_CONFIG_PATH.parent / "_refused_export_never_created"
    with pytest.raises(build.BuildError, match="inside the repository"):
        exported(ranked, inside)
    assert not inside.exists()


def test_export_needs_a_complete_ranking_with_matching_files_config_and_snapshot(ranked, tmp_path):
    snapshot, run = ranked
    limited = rank.write_ranking(snapshot.out_dir, tmp_path / "limited", limit=2)
    with pytest.raises(export.ExportError, match="complete ranking"):
        export.export(limited.out_dir, snapshot.out_dir, tmp_path / "out", approval_reference=APPROVAL)
    tampered = tmp_path / "tampered"
    shutil.copytree(run.out_dir, tampered)
    (tampered / rank.RANKED_FILE).write_bytes((tampered / rank.RANKED_FILE).read_bytes() + b"\0")
    with pytest.raises(export.ExportError, match="does not match the rank manifest"):
        export.export(tampered, snapshot.out_dir, tmp_path / "out", approval_reference=APPROVAL)
    document = json.loads(rank.RANK_CONFIG_PATH.read_text(encoding="utf-8"))
    document["version"] = document["version"] + "-synthetic"
    other_config = tmp_path / "rank_config.json"
    other_config.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(export.ExportError, match="rank config SHA-256"):
        exported(ranked, tmp_path / "out", config=other_config)
    other = build.build(["TX"], out_dir=tmp_path / "other", cache=seeded_cache(tmp_path / "raw", manual=False))
    with pytest.raises(export.ExportError, match="another snapshot"):
        export.export(run.out_dir, other.out_dir, tmp_path / "out", approval_reference=APPROVAL)
    for approval in ("PENDING-owner-review", " ", "café"):
        with pytest.raises(export.ExportError, match="approval"):
            export.export(run.out_dir, snapshot.out_dir, tmp_path / "out", approval_reference=approval)
    with pytest.raises(export.ExportError, match="--top"):
        exported(ranked, tmp_path / "out", top=0)
    assert not (tmp_path / "out").exists()


def test_export_fails_closed_on_a_license_outside_the_runtime_allowlist(ranked, tmp_path):
    snapshot, run = ranked
    copy = tmp_path / "snap"
    shutil.copytree(snapshot.out_dir, copy)
    manifest = json.loads((copy / build.MANIFEST_FILE).read_text(encoding="utf-8"))
    manifest["license_union"].append({"id": "CC-BY-NC-4.0", "source_id": "synthetic_restricted"})
    (copy / build.MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(export.ExportError, match="not among the reviewed"):
        export.export(run.out_dir, copy, tmp_path / "out", approval_reference=APPROVAL)
    assert not (tmp_path / "out").exists()


def test_previous_snapshot_counts_new_sites(ranked, tmp_path):
    snapshot, _ = ranked
    previous = build.build(["TX"], out_dir=tmp_path / "previous", cache=seeded_cache(tmp_path / "raw", manual=False))
    exported(ranked, tmp_path / "out", previous_snapshot=previous.out_dir)
    manifest = runtime.load_export((tmp_path / "out" / runtime.OBJECT_NAME).read_bytes())["manifest"]
    current = {site["site_id"] for site in rank.load_snapshot(snapshot.out_dir).sites}
    earlier = {site["site_id"] for site in rank.load_snapshot(previous.out_dir).sites}
    assert manifest["previous_snapshot_id"] == previous.snapshot_id
    assert manifest["new_sites"] == len(current - earlier) > 0


def test_lead_capability_reads_the_capability_fit_basis_without_changing_ranker_output(ranked, tmp_path):
    snapshot, run = ranked
    before = run.ranked_path.read_bytes()
    assert {rank.lead_capability(row) for row in ranked_rows(run)} <= set(rank.load_config().capability_weights)
    row = {"site_id": "x", "capabilities": ["kitting_assembly", "palletizing_depalletizing"],
           "components": {"capability_fit": {"basis": "kitting_assembly via secondary site type warehouse_general"}}}
    assert rank.lead_capability(row) == "kitting_assembly"
    for bad in ({**row, "components": {"capability_fit": {"basis": "no capability in scope"}}},
                {**row, "capabilities": ["palletizing_depalletizing"]},
                {**row, "capabilities": ["synthetic_unweighted"],
                 "components": {"capability_fit": {"basis": "synthetic_unweighted"}}}, {}, "not a row"):
        with pytest.raises(rank.RankError):
            rank.lead_capability(bad)
    again = rank.write_ranking(snapshot.out_dir, tmp_path / "again")
    assert again.ranked_path.read_bytes() == before == run.ranked_path.read_bytes()


def test_cli_export_prints_counts_and_ids_but_no_site_names(ranked, tmp_path, capsys):
    snapshot, run = ranked
    base = ["export", "--ranking", str(run.out_dir), "--snapshot", str(snapshot.out_dir), "--out", str(tmp_path / "cli")]
    assert main(base + ["--approval-reference", APPROVAL]) == 0
    printed = capsys.readouterr().out
    summary = json.loads(printed)
    rows = runtime.load_export((tmp_path / "cli" / runtime.OBJECT_NAME).read_bytes())["rows"]
    assert summary["rows"] == len(rows) and summary["uri"].endswith("/" + runtime.OBJECT_NAME)
    names = {row["name"] for row in rows if row["name"]}
    assert names and not any(name in printed for name in names)
    assert main(base + ["--approval-reference", "PENDING-owner"]) == 1
    assert "approval" in capsys.readouterr().err
