import unittest

from backend.engine.validation.csp import BuildCSP


class TestBuildCSP(unittest.TestCase):
    def setUp(self):
        self.validator = BuildCSP()

    def test_valid_build(self):
        build_data = {
            "main_skill": {"links": 6},
            "items": [{"requirements": {"str": 100}, "sockets": ["R", "R", "R", "R"]}],
            "attributes": {"str": 150, "dex": 50, "int": 50},
            "reservation": 800,
            "total_mana": 1000,
            "passive_tree": {
                "allocated": [1, 2, 3],
                "adjacencies": {1: [2], 2: [1, 3], 3: [2]},
                "start_nodes": [1],
            },
        }
        is_valid, errors = self.validator.validate(build_data)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_invalid_links(self):
        build_data = {
            "main_skill": {"links": 5},
            "items": [],
            "attributes": {"str": 100, "dex": 100, "int": 100},
            "reservation": 0,
            "total_mana": 100,
            "passive_tree": {"allocated": [], "adjacencies": {}, "start_nodes": []},
        }
        is_valid, errors = self.validator.validate(build_data)
        self.assertFalse(is_valid)
        self.assertIn("Main skill must have 6 links", errors)

    def test_invalid_requirements(self):
        build_data = {
            "main_skill": {"links": 6},
            "items": [{"requirements": {"str": 200}, "sockets": []}],
            "attributes": {"str": 150, "dex": 50, "int": 50},
            "reservation": 0,
            "total_mana": 100,
            "passive_tree": {"allocated": [], "adjacencies": {}, "start_nodes": []},
        }
        is_valid, errors = self.validator.validate(build_data)
        self.assertFalse(is_valid)
        self.assertTrue(any("requires 200 str" in e.lower() for e in errors))

    def test_invalid_reservation(self):
        build_data = {
            "main_skill": {"links": 6},
            "items": [],
            "attributes": {"str": 100, "dex": 100, "int": 100},
            "reservation": 1100,
            "total_mana": 1000,
            "passive_tree": {"allocated": [], "adjacencies": {}, "start_nodes": []},
        }
        is_valid, errors = self.validator.validate(build_data)
        self.assertFalse(is_valid)
        self.assertIn("Mana reservation exceeds total mana", errors)

    def test_invalid_passive_connectivity(self):
        build_data = {
            "main_skill": {"links": 6},
            "items": [],
            "attributes": {"str": 100, "dex": 100, "int": 100},
            "reservation": 0,
            "total_mana": 100,
            "passive_tree": {
                "allocated": [1, 2, 10],
                "adjacencies": {1: [2], 2: [1], 10: []},
                "start_nodes": [1],
            },
        }
        is_valid, errors = self.validator.validate(build_data)
        self.assertFalse(is_valid)
        self.assertIn("Passive tree is not connected to a starting node", errors)

    def test_invalid_socket_colors(self):
        build_data = {
            "main_skill": {"links": 6},
            "items": [
                {"requirements": {"str": 0, "dex": 0, "int": 0}, "sockets": ["B", "B", "B", "B"]}
            ],
            "attributes": {"str": 100, "dex": 100, "int": 100},
            "reservation": 0,
            "total_mana": 100,
            "passive_tree": {"allocated": [], "adjacencies": {}, "start_nodes": []},
        }
        is_valid, errors = self.validator.validate(build_data)
        self.assertFalse(is_valid)
        self.assertTrue(any("too many B sockets" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
