from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lora_mesh_sim.simulation import SofiaMeshSimulation


class MQTTValidityTest(unittest.TestCase):
    def test_small_network_initializes_and_steps(self) -> None:
        simulation = SofiaMeshSimulation(node_count=12, seed=3)
        try:
            initial_snapshot = simulation.last_snapshot
            self.assertIsNotNone(simulation.runtime)
            self.assertIsNotNone(initial_snapshot)
            self.assertGreaterEqual(initial_snapshot.metrics["relays"], 1)

            stepped_snapshot = simulation.step()
            self.assertEqual(stepped_snapshot.metrics["nodes"], 12)
            self.assertGreaterEqual(stepped_snapshot.metrics["total_sent"], 0)
            self.assertGreaterEqual(stepped_snapshot.metrics["total_delivered"], 0)
            self.assertGreaterEqual(stepped_snapshot.metrics["total_dropped"], 0)
        finally:
            simulation.shutdown()


if __name__ == "__main__":
    unittest.main()
