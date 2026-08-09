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
            "raw_status_normalized": "running",
        }
    ]
