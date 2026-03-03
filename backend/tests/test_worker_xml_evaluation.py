from __future__ import annotations

import json

from backend.tests.test_api_flow import _import_payload, _prepare_client


def _player_stats_xml() -> str:
    return """
<PathOfBuilding>
  <Build level="100" className="Witch" ascendClassName="Occultist" mainSocketGroup="1"/>
  <PlayerStat stat="FullDPS" value="250000"/>
  <PlayerStat stat="MaximumHitTaken" value="9000"/>
  <PlayerStat stat="Life" value="5200"/>
  <PlayerStat stat="Mana" value="1200"/>
  <PlayerStat stat="Armour" value="18000"/>
  <PlayerStat stat="Evasion" value="12000"/>
  <PlayerStat stat="EnergyShield" value="1500"/>
  <PlayerStat stat="FireResist" value="75"/>
  <PlayerStat stat="ColdResist" value="75"/>
  <PlayerStat stat="LightningResist" value="75"/>
  <PlayerStat stat="ChaosResist" value="30"/>
  <PlayerStat stat="Str" value="140"/>
  <PlayerStat stat="Dex" value="110"/>
  <PlayerStat stat="Int" value="280"/>
  <PlayerStat stat="LifeUnreservedPercent" value="100"/>
  <PlayerStat stat="ManaUnreservedPercent" value="45"/>
  <PlayerStat stat="EffectiveMovementSpeedMod" value="1.2"/>
  <PlayerStat stat="BlockChance" value="20"/>
  <PlayerStat stat="SpellBlockChance" value="15"/>
</PathOfBuilding>
""".strip()


def test_evaluate_uses_worker_player_stats_when_enabled(tmp_path) -> None:
    client, _repo = _prepare_client(tmp_path)
    try:
        payload = _import_payload(profile_id="mapping", ruleset_id="standard")
        payload["xml"] = _player_stats_xml()

        import_response = client.post("/import", json=payload)
        assert import_response.status_code == 201
        build_id = import_response.json()["build_id"]

        evaluate_response = client.post(f"/evaluate/{build_id}")
        assert evaluate_response.status_code == 200
        body = evaluate_response.json()
        assert body["status"] in {"evaluated", "failed"}
        assert body["scenario_results"], "scenario results should not be empty"
        first_metric = body["scenario_results"][0]
        assert first_metric["full_dps"] > 0
        assert first_metric["max_hit"] > 0

        metrics_raw_path = tmp_path / "data" / "builds" / build_id / "metrics_raw.json"
        assert metrics_raw_path.exists()
        metrics_raw = json.loads(metrics_raw_path.read_text())
        mapping_payload = metrics_raw.get("mapping_t16")
        assert mapping_payload is not None
        assert mapping_payload["source"] == "pob_xml_playerstats"
        assert mapping_payload["metrics_source"] == "fallback"
    finally:
        client.close()


def test_evaluate_missing_player_stats_returns_missing_metrics(tmp_path) -> None:
    client, _repo = _prepare_client(tmp_path)
    try:
        payload = _import_payload(profile_id="mapping", ruleset_id="standard")
        payload["xml"] = "<build><id>sample</id></build>"

        import_response = client.post("/import", json=payload)
        assert import_response.status_code == 201
        build_id = import_response.json()["build_id"]

        evaluate_response = client.post(f"/evaluate/{build_id}")
        assert evaluate_response.status_code in {400, 502}
        body = evaluate_response.json()
        assert body["error"]["code"] in {
            "missing_metrics",
            "worker_unavailable",
            "worker_response_error",
        }
        assert "ENABLE_WORKER_EVAL=true" not in json.dumps(body)
    finally:
        client.close()
