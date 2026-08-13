from blueprint_pipeline import vast_provider_adapter as vpa


def test_single_instance_response_is_not_split_at_nested_provider_fields() -> None:
    row = {
        "id": 22,
        "actual_status": "running",
        "ports": {"22/tcp": [{"HostPort": "12345"}]},
        "search": {"gpu_name": "L40S"},
    }

    assert vpa._instance_list_rows({"instances": row}) == [row]
    assert vpa._active_instance_rows_from_payload({"instances": row}) == [
        {
            "id": 22,
            "machine_id": None,
            "has_avx": None,
            "gpu_name": None,
            "actual_status": "running",
            "cur_state": None,
            "status": None,
            "intended_status": None,
            "dph_total": None,
            # How an instance says which lane created it. Without it the
            # prelaunch guard can only ask "is anything running", never "is
            # anything of *mine* running", and a concurrent operator's instance
            # is indistinguishable from an orphan of ours (#473).
            "label": None,
            "raw_status_normalized": "running",
        }
    ]


def test_the_sanitized_row_carries_the_label_that_identifies_its_owner() -> None:
    """The guard scopes itself by label prefix, so the label has to survive."""

    rows = vpa._active_instance_rows_from_payload(
        {
            "instances": {
                "id": 47605330,
                "actual_status": "running",
                "label": "blueprint-adp-content-agents-run-1",
            }
        }
    )

    assert [row["label"] for row in rows] == ["blueprint-adp-content-agents-run-1"]
