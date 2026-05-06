"""
Scenario Loader
===============
Load and manage demo scenarios from YAML configuration.
"""
from pathlib import Path
from typing import Dict, List, Optional

import yaml


class ScenarioLoader:
    """Load scenarios from YAML configuration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "scenarios.yaml"
        
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.scenarios = self.config["scenarios"]
    
    def list_scenarios(self) -> List[Dict]:
        """List all available scenarios."""
        return [
            {
                "id": data["id"],
                "name": data["name"],
                "description": data.get("description", ""),
                "situation_id": data["situation_id"]
            }
            for key, data in self.scenarios.items()
        ]
    
    def get_scenario(self, scenario_id: str) -> Optional[Dict]:
        """Get scenario by ID or key."""
        # Try by ID first
        for key, data in self.scenarios.items():
            if data["id"] == scenario_id:
                return data
        
        # Try by key
        return self.scenarios.get(scenario_id)
    
    def display_menu(self) -> str:
        """Generate a menu of scenarios."""
        lines = ["Select a scenario:"]
        for key, data in self.scenarios.items():
            lines.append(f"{data['id']}. {data['name']} ({data['situation_id']})")
        lines.append("")
        return "\n".join(lines)


# Convenience function
def load_scenarios(config_path: Optional[Path] = None) -> ScenarioLoader:
    """Load scenarios from configuration."""
    return ScenarioLoader(config_path)
