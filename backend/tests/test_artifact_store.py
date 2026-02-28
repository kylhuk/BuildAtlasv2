from __future__ import annotations

import pytest

from backend.engine.artifacts.store import (
    canonical_xml_hash,
    format_ruleset_id,
    parse_ruleset_id,
    read_build_artifacts,
    write_build_artifacts,
    write_build_constraints,
)


def test_write_read_round_trip(tmp_path):
    xml = "<build>\n  <item>PoB</item>\n</build>"
    code = "print('artifact')"
    genome = {"layers": ["core"], "version": 1}
    scenarios = [{"id": "shop"}]
    metrics = {"score": 100}
    build_details = {
        "schema_version": 1,
        "identity": {"class": "Ranger", "ascendancy": "Deadeye", "main_skill": "rain_of_arrows"},
        "items": {"slot_templates": [{"slot_id": "weapon_1", "base_type": "Bow"}]},
        "passives": {"node_ids": [123, 456]},
        "gems": {"groups": [{"id": "main", "gems": ["rain_of_arrows", "mirage_archer"]}]},
        "exports": {"xml_available": True, "code_available": True},
    }

    provenance = write_build_artifacts(
        "build-42",
        xml=xml,
        code=code,
        genome=genome,
        scenarios_used=scenarios,
        raw_metrics=metrics,
        build_details=build_details,
        base_path=tmp_path,
    )

    assert provenance.xml_hash == canonical_xml_hash(xml)

    artifacts = read_build_artifacts("build-42", base_path=tmp_path)
    assert artifacts.xml == xml
    assert artifacts.code == code
    assert artifacts.genome == genome
    assert artifacts.scenarios_used == scenarios
    assert artifacts.raw_metrics == metrics
    assert artifacts.build_details == build_details

    tmp_files = [part for part in provenance.paths.base_dir.iterdir() if part.name.endswith(".tmp")]
    assert not tmp_files


def test_write_build_constraints_round_trip(tmp_path):
    payload = {
        "schema_version": 1,
        "status": "pass",
    }
    write_build_artifacts(
        "build-42",
        xml=None,
        code="",
        base_path=tmp_path,
    )
    constraints_path = write_build_constraints(
        "build-42",
        payload,
        base_path=tmp_path,
    )
    assert constraints_path.exists()

    artifacts = read_build_artifacts(
        "build-42",
        base_path=tmp_path,
    )
    assert artifacts.constraints == payload


def test_canonical_xml_hash_normalizes_newlines():
    xml_lf = "<root>\n<value>1</value>\n</root>"
    xml_crlf = xml_lf.replace("\n", "\r\n")
    assert canonical_xml_hash(xml_lf) == canonical_xml_hash(xml_crlf)


def test_canonical_xml_hash_ignores_formatting_and_attribute_order():
    xml_base = "<root><item attr2='two' attr1='one'>value</item></root>"
    xml_variant = "<root>\n  <item attr1='one' attr2='two'>value</item>\n</root>"
    assert canonical_xml_hash(xml_base) == canonical_xml_hash(xml_variant)


def test_canonical_xml_hash_detects_semantic_changes():
    xml_base = "<root><item attr='one'>value</item></root>"
    xml_changed = "<root><item attr='two'>value</item></root>"
    assert canonical_xml_hash(xml_base) != canonical_xml_hash(xml_changed)


def test_build_id_validation_rejects_traversal(tmp_path):
    invalid_ids = [
        "..",
        "../escape",
        "/absolute",
        "build/dir",
        "build\\dir",
        "build..",
        "unsafe:char",
        "build name",
    ]
    for build_id in invalid_ids:
        with pytest.raises(ValueError):
            write_build_artifacts(build_id, xml="<root/>", code="code", base_path=tmp_path)
        with pytest.raises(ValueError):
            read_build_artifacts(build_id, base_path=tmp_path)


def test_ruleset_format_and_parse():
    ruleset = format_ruleset_id("abc123", "2025.2", "prices-v1")
    assert ruleset == "pob:abc123|scenarios:2025.2|prices:prices-v1"

    commit, scenarios, prices = parse_ruleset_id(ruleset)
    assert commit == "abc123"
    assert scenarios == "2025.2"
    assert prices == "prices-v1"


def test_ruleset_invalid_inputs():
    with pytest.raises(ValueError):
        format_ruleset_id("", "2025.2", "prices-v1")

    with pytest.raises(ValueError):
        format_ruleset_id("abc", "bad|token", "prices")

    with pytest.raises(ValueError):
        parse_ruleset_id("pob:abc|scenarios")
