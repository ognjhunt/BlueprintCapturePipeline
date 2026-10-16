"""The staging command must preserve scored cells and the frozen recovery identity."""
import pytest

from tests.test_team_candidate_episode_recovery import scored, refused


def test_resume_layout_adopts_scores_and_places_refused_cell_first():
    from scripts.stage_internal_team_evaluation_resume import remaining_groups

    cells = [scored(str(index)) for index in range(200)]
    retained = cells[:25] + cells[50:74] + [refused("74")]
    groups, refused_ids = remaining_groups(cells, retained)
    assert [len(group) for group in groups] == [25, 25, 25, 25, 25, 25, 1]
    assert groups[0][0]["cell_id"] == "74" and refused_ids == ["74"]
    assert len({cell["cell_id"] for group in groups for cell in group}) == 151
    with pytest.raises(ValueError, match="not_unique"):
        remaining_groups(cells, retained + [cells[0]])

