import gzip
from pathlib import Path

from backend.engine.pricing.costs import extract_build_cost_requirements


def _write_gzip_xml(build_dir: Path, xml_content: str) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(build_dir / "build.xml.gz", "wb") as handle:
        handle.write(xml_content.encode("utf-8"))


def _write_code_xml(build_dir: Path, xml_content: str) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "build.code.txt").write_text(xml_content, encoding="utf-8")


def test_extracts_uniques_and_gems_from_gzip(tmp_path: Path) -> None:
    build_id = "cost-slice"
    build_dir = tmp_path / "data" / "builds" / build_id
    gzip_xml = """
    <build>
      <items>
        <item slot="Body">
          <rarity>Unique</rarity>
          <name>Headhunter</name>
        </item>
        <item>
          <slot>Weapon</slot>
          <rarity>unique</rarity>
          <Name>Lion's Roar</Name>
        </item>
        <item slot="Gloves">
          <rarity>Rare</rarity>
          <name>Common Gloves</name>
        </item>
      </items>
      <gems>
        <skillGem>
          <name>Frostbolt</name>
          <level>20</level>
          <quality>23</quality>
        </skillGem>
        <skillGem>
          <name>Ruthless Support</name>
          <level>1</level>
        </skillGem>
      </gems>
    </build>
    """
    _write_gzip_xml(build_dir, gzip_xml)
    code_xml = """
    <build>
      <items>
        <item slot="Gloves">
          <rarity>unique</rarity>
          <name>Arc</name>
        </item>
      </items>
      <gems>
        <skillGem>
          <name>Arc</name>
        </skillGem>
      </gems>
    </build>
    """
    _write_code_xml(build_dir, code_xml)

    result = extract_build_cost_requirements(build_id, base_path=tmp_path)
    assert {item.name for item in result.unique_items} == {
        "Headhunter",
        "Lion's Roar",
    }
    gem_map = {gem.name: gem for gem in result.skill_gems}
    assert gem_map["Frostbolt"].level == 20
    assert gem_map["Frostbolt"].quality == 23
    assert gem_map["Ruthless Support"].level == 1
    assert gem_map["Ruthless Support"].quality is None

    (build_dir / "build.xml.gz").unlink()
    fallback = extract_build_cost_requirements(build_id, base_path=tmp_path)
    assert len(fallback.unique_items) == 1
    assert fallback.unique_items[0].name == "Arc"
    assert fallback.skill_gems[0].name == "Arc"
