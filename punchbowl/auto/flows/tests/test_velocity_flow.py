import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from punchbowl.auto.control.db import File
from punchbowl.auto.flows.velocity import (
    TRACK_VELOCITY_DEFAULTS,
    level3_VAM_construct_file_info,
    level3_VAM_construct_flow_info,
    level3_VAM_query_ready_files,
)
from punchbowl.level3.velocity import get_cadence_min, get_required_frame_count

T0 = datetime(2026, 1, 1, 0, 0, 0)


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    # Only the files table is needed; the flows table uses MySQL-only column types
    File.__table__.create(engine)
    with Session(engine) as session:
        yield session


def _pipeline_config(**l3_vam_config):
    return {"file_version": "1", "flows": {"level3_VAM": {"t0": T0.isoformat(), **l3_vam_config}}}


def _add_files(session, product_code, start, end, skip=()):
    """Add files at the product cadence from start to end, except the frame indices in skip."""
    cadence = timedelta(minutes=get_cadence_min(product_code))
    t, i = start, 0
    while t < end:
        if i not in skip:
            session.add(File(level="3", file_type=product_code[:2], observatory=product_code[2], state="created",
                             file_version="1", software_version="none", polarization="C", date_obs=t))
        t += cadence
        i += 1
    session.commit()


def _n_required(product_code):
    n_required, _, _ = get_required_frame_count(product_code, TRACK_VELOCITY_DEFAULTS["flow_window_hours"],
                                                TRACK_VELOCITY_DEFAULTS["time_avg_hours"])
    return n_required


# Frame times are offset from the nominal reference times, as for actual observations
OFFSET = timedelta(minutes=1, seconds=13)


def test_groups_at_frames_nearest_to_nominal_reference_times(session):
    _add_files(session, "PTM", T0 + OFFSET, T0 + timedelta(hours=36))
    groups = level3_VAM_query_ready_files.fn(session, _pipeline_config())

    reference_times = [g.reference_time for g in groups]
    # 12-hour inputs centered on 09:00, 15:00, 21:00 and 03:00 the next day fit in the 36 hours of data
    nominal_times = [T0 + timedelta(hours=h) for h in (27, 21, 15, 9)]
    assert len(groups) == len(nominal_times)
    for ref_time, nominal_time, group in zip(reference_times, nominal_times, groups):
        # The reference time is the observation time of the frame nearest to the nominal time
        assert ref_time - nominal_time == OFFSET
        assert ref_time in [f.date_obs for f in group]
        assert len(group) == _n_required("PTM")
    # Consecutive input windows overlap, so some files feed more than one map
    assert set(groups[0]) & set(groups[1])


def test_existing_velocity_maps_are_skipped(session):
    _add_files(session, "PTM", T0 + OFFSET, T0 + timedelta(hours=36))
    groups = level3_VAM_query_ready_files.fn(session, _pipeline_config())
    skipped = groups[0].reference_time
    session.add(File(level="3", file_type="VA", observatory="M", state="created", file_version="1",
                     software_version="none", polarization="Y", date_obs=skipped))
    session.commit()

    new_groups = level3_VAM_query_ready_files.fn(session, _pipeline_config())
    assert skipped not in [g.reference_time for g in new_groups]
    assert len(new_groups) == len(groups) - 1


def test_tolerated_missing_frames_are_scheduled(session):
    # Two consecutive missing PTM frames just before 15:00, in the flow window of that map
    i_15 = round((timedelta(hours=15) - OFFSET) / timedelta(minutes=get_cadence_min("PTM")))
    _add_files(session, "PTM", T0 + OFFSET, T0 + timedelta(hours=36), skip=(i_15 - 2, i_15 - 1))
    groups = level3_VAM_query_ready_files.fn(session, _pipeline_config())
    group = next(g for g in groups if g.reference_time == T0 + timedelta(hours=15) + OFFSET)
    assert len(group) == _n_required("PTM") - 2


def test_too_many_missing_frames_are_not_scheduled(session):
    # Three consecutive missing PTM frames just before 15:00, in the flow window of that map
    i_15 = round((timedelta(hours=15) - OFFSET) / timedelta(minutes=get_cadence_min("PTM")))
    _add_files(session, "PTM", T0 + OFFSET, T0 + timedelta(hours=36), skip=(i_15 - 3, i_15 - 2, i_15 - 1))
    groups = level3_VAM_query_ready_files.fn(session, _pipeline_config())
    assert T0 + timedelta(hours=15) + OFFSET not in [g.reference_time for g in groups]
    # The maps whose flow windows do not hold the missing frames are still scheduled
    assert T0 + timedelta(hours=27) + OFFSET in [g.reference_time for g in groups]


def test_product_code_from_config_selects_input_files(session):
    _add_files(session, "PTM", T0, T0 + timedelta(hours=36))
    _add_files(session, "CAM", T0, T0 + timedelta(hours=36))
    groups = level3_VAM_query_ready_files.fn(session, _pipeline_config(product_code="CAM"))
    assert groups
    assert all(f.file_type == "CA" and f.observatory == "M" for g in groups for f in g)


def test_unsupported_product_code_raises(session):
    _add_files(session, "PTM", T0, T0 + timedelta(hours=36))
    with pytest.raises(ValueError):
        level3_VAM_query_ready_files.fn(session, _pipeline_config(product_code="PIM"))


def test_flow_and_file_info_use_group_reference_time(session):
    _add_files(session, "PTM", T0, T0 + timedelta(hours=36))
    pipeline_config = _pipeline_config()
    group = level3_VAM_query_ready_files.fn(session, pipeline_config)[0]

    output_file = level3_VAM_construct_file_info.fn(group, pipeline_config, reference_time=None)[0]
    assert output_file.date_obs == group.reference_time

    flow = level3_VAM_construct_flow_info.fn(group, output_file, pipeline_config, reference_time=None)
    call_data = json.loads(flow.call_data)
    assert datetime.fromisoformat(call_data["reference_time"]) == group.reference_time
    assert call_data["product_code"] == "PTM"
    assert len(call_data["files"]) == len(group)
    # With nothing set in the pipeline configuration, all parameters take the track_velocity defaults
    for key, default in TRACK_VELOCITY_DEFAULTS.items():
        expected = list(default) if isinstance(default, tuple) else default
        assert call_data[key] == expected


def test_flow_parameters_from_config_override_defaults(session):
    _add_files(session, "PTM", T0, T0 + timedelta(hours=36))
    pipeline_config = _pipeline_config(delta_t=3, annuli_centers_rs=[20, 40, 60])
    group = level3_VAM_query_ready_files.fn(session, pipeline_config)[0]
    output_file = level3_VAM_construct_file_info.fn(group, pipeline_config)[0]
    call_data = json.loads(level3_VAM_construct_flow_info.fn(group, output_file, pipeline_config).call_data)
    assert call_data["delta_t"] == 3
    assert call_data["annuli_centers_rs"] == [20, 40, 60]
