from __future__ import annotations

from backend.tests.test_api_flow import _import_payload, _prepare_client, _share_import_payload


def _xml_with_composition() -> str:
    return """
<PathOfBuilding>
  <Build className="Witch" ascendClassName="Occultist"/>
  <Tree activeSpec="1">
    <Spec title="1" nodes="123,456,789"/>
  </Tree>
  <Items>
    <Item slot="Weapon 1">Rarity: Unique\nDuskdawn\nMaelstrom Staff</Item>
    <Item slot="Body Armour">Rarity: Rare\nViper Cloak</Item>
  </Items>
  <Skills>
    <Skill slot="Weapon 1">
      <Gem skillId="Arc" level="20" quality="20" enabled="true"/>
      <Gem skillId="AddedLightningDamageSupport" level="20" quality="20" enabled="true"/>
    </Skill>
  </Skills>
</PathOfBuilding>
""".strip()


def test_import_infers_identity_and_returns_build_details(tmp_path) -> None:
    client, _repo = _prepare_client(tmp_path)
    try:
        payload = _import_payload(profile_id="mapping", ruleset_id="standard")
        payload["xml"] = _xml_with_composition()
        payload["metadata"]["class"] = "unknown"
        payload["metadata"]["ascendancy"] = "unknown"
        payload["metadata"]["main_skill"] = "unknown"

        import_response = client.post("/import", json=payload)
        assert import_response.status_code == 201
        build_id = import_response.json()["build_id"]

        detail_response = client.get(f"/builds/{build_id}")
        assert detail_response.status_code == 200
        detail = detail_response.json()
        build = detail["build"]
        assert build["class"] == "Witch"
        assert build["ascendancy"] == "Occultist"
        assert build["main_skill"] == "Arc"

        build_details = detail.get("build_details")
        assert isinstance(build_details, dict)
        assert build_details.get("source") == "xml"

        identity = build_details.get("identity")
        assert isinstance(identity, dict)
        assert identity.get("class") == "Witch"
        assert identity.get("ascendancy") == "Occultist"
        assert identity.get("main_skill") == "Arc"

        items = build_details.get("items")
        assert isinstance(items, dict)
        assert len(items.get("items", [])) >= 1

        passives = build_details.get("passives")
        assert isinstance(passives, dict)
        assert passives.get("node_ids") == ["123", "456", "789"]

        gems = build_details.get("gems")
        assert isinstance(gems, dict)
        groups = gems.get("groups")
        assert isinstance(groups, list) and groups
        first_group = groups[0]
        assert first_group.get("link_count") == 2
    finally:
        client.close()


def test_share_code_import_build_details_use_share_source(tmp_path) -> None:
    client, _repo = _prepare_client(tmp_path)
    try:
        payload = _share_import_payload()
        import_response = client.post("/import", json=payload)
        assert import_response.status_code == 201
        build_id = import_response.json()["build_id"]

        detail_response = client.get(f"/builds/{build_id}")
        assert detail_response.status_code == 200
        detail = detail_response.json()
        build_details = detail.get("build_details")
        assert isinstance(build_details, dict)
        assert build_details.get("source") == "share_code"

        items = build_details.get("items")
        gems = build_details.get("gems")
        passives = build_details.get("passives")
        assert isinstance(items, dict) and items.get("items") == []
        assert isinstance(gems, dict) and gems.get("groups") == []
        assert isinstance(passives, dict) and passives.get("node_ids") == []

        exports = build_details.get("exports")
        assert isinstance(exports, dict)
        assert exports.get("share_code_available") is True
    finally:
        client.close()
