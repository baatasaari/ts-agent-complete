"""
Configuration Loader
====================
Loads all FCA PS25/22 regulatory data from YAML files.

This module provides a centralized way to load:
- Situations (fca_ts_situations.yaml)
- Segmentations (fca_ts_segmentations.yaml) 
- Suggestions (fca_ts_suggestions.yaml)
- Compliance Checks (fca_ts_compliance_checks.yaml)

All application logic should load from these YAML files rather than
hardcoding regulatory data in Python.
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import yaml


@dataclass
class Situation:
    """A financial support need or objective (COBS 9B.3)."""
    situation_id: str
    name: str
    description: str
    in_scope_products: List[str]
    intent_id: str
    characteristics: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Situation':
        return cls(
            situation_id=data['situation_id'],
            name=data['name'],
            description=data.get('description', ''),
            in_scope_products=data.get('in_scope_products', []),
            intent_id=data.get('intent_id', ''),
            characteristics=data.get('characteristics', {})
        )


@dataclass
class Segmentation:
    """Consumer segmentation for targeted support."""
    segment_id: str
    name: str
    description: str
    situation_id: str
    criteria: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Segmentation':
        return cls(
            segment_id=data['segment_id'],
            name=data['name'],
            description=data.get('description', ''),
            situation_id=data.get('situation_id', ''),
            criteria=data.get('criteria', {})
        )


@dataclass
class Suggestion:
    """Ready-made suggestion for a segment."""
    suggestion_id: str
    segment_id: str
    name: str
    description: str
    product_type: str
    compliance_checks: List[str]
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Suggestion':
        return cls(
            suggestion_id=data['suggestion_id'],
            segment_id=data['segment_id'],
            name=data['name'],
            description=data.get('description', ''),
            product_type=data.get('product_type', ''),
            compliance_checks=data.get('compliance_checks', [])
        )


@dataclass
class ComplianceCheck:
    """PS25/22 compliance check definition."""
    check_id: str
    name: str
    phase: str
    severity: str
    description: str
    rulebook_ref: str
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ComplianceCheck':
        return cls(
            check_id=data['check_id'],
            name=data['name'],
            phase=data.get('phase', 'PDC'),
            severity=data.get('severity', 'HARD'),
            description=data.get('description', ''),
            rulebook_ref=data.get('rulebook_ref', '')
        )


class ConfigLoader:
    """
    Central configuration loader for FCA PS25/22 regulatory data.
    
    Usage:
        config = ConfigLoader()
        situations = config.load_situations()
        segments = config.load_segments()
        suggestions = config.load_suggestions()
        checks = config.load_compliance_checks()
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize config loader.
        
        Args:
            config_dir: Path to config directory. If None, uses default location.
        """
        if config_dir is None:
            # Default: project_root/config/
            config_dir = Path(__file__).parent.parent.parent / "config"
        
        self.config_dir = Path(config_dir)
        self._situations_cache = None
        self._segments_cache = None
        self._suggestions_cache = None
        self._checks_cache = None
    
    def _load_yaml(self, filename: str) -> Dict:
        """Load a YAML file and return parsed data."""
        filepath = self.config_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    
    def load_situations(self, force_reload: bool = False) -> Dict[str, Situation]:
        """
        Load all situations from fca_ts_situations.yaml.
        
        Args:
            force_reload: Force reload from disk (bypass cache)
            
        Returns:
            Dictionary mapping situation_id to Situation object
        """
        if self._situations_cache is None or force_reload:
            data = self._load_yaml('fca_ts_situations.yaml')
            self._situations_cache = {
                sit['situation_id']: Situation.from_dict(sit)
                for sit in data.get('situations', [])
            }
        return self._situations_cache
    
    def load_segments(self, force_reload: bool = False) -> Dict[str, Segmentation]:
        """
        Load all segments from fca_ts_segmentations.yaml.
        
        Args:
            force_reload: Force reload from disk (bypass cache)
            
        Returns:
            Dictionary mapping segment_id to Segmentation object
        """
        if self._segments_cache is None or force_reload:
            data = self._load_yaml('fca_ts_segmentations.yaml')
            self._segments_cache = {
                seg['segment_id']: Segmentation.from_dict(seg)
                for seg in data.get('segmentations', [])
            }
        return self._segments_cache
    
    def load_suggestions(self, force_reload: bool = False) -> Dict[str, Suggestion]:
        """
        Load all suggestions from fca_ts_suggestions.yaml.
        
        Args:
            force_reload: Force reload from disk (bypass cache)
            
        Returns:
            Dictionary mapping suggestion_id to Suggestion object
        """
        if self._suggestions_cache is None or force_reload:
            data = self._load_yaml('fca_ts_suggestions.yaml')
            self._suggestions_cache = {
                sug['suggestion_id']: Suggestion.from_dict(sug)
                for sug in data.get('suggestions', [])
            }
        return self._suggestions_cache
    
    def load_compliance_checks(self, force_reload: bool = False) -> Dict[str, ComplianceCheck]:
        """
        Load all compliance checks from fca_ts_compliance_checks.yaml.
        
        Args:
            force_reload: Force reload from disk (bypass cache)
            
        Returns:
            Dictionary mapping check_id to ComplianceCheck object
        """
        if self._checks_cache is None or force_reload:
            data = self._load_yaml('fca_ts_compliance_checks.yaml')
            self._checks_cache = {
                check['check_id']: ComplianceCheck.from_dict(check)
                for check in data.get('compliance_checks', [])
            }
        return self._checks_cache
    
    def get_situation(self, situation_id: str) -> Optional[Situation]:
        """Get a specific situation by ID."""
        return self.load_situations().get(situation_id)
    
    def get_segment(self, segment_id: str) -> Optional[Segmentation]:
        """Get a specific segment by ID."""
        return self.load_segments().get(segment_id)
    
    def get_suggestion(self, suggestion_id: str) -> Optional[Suggestion]:
        """Get a specific suggestion by ID."""
        return self.load_suggestions().get(suggestion_id)
    
    def get_check(self, check_id: str) -> Optional[ComplianceCheck]:
        """Get a specific compliance check by ID."""
        return self.load_compliance_checks().get(check_id)
    
    def get_segments_for_situation(self, situation_id: str) -> List[Segmentation]:
        """Get all segments for a given situation."""
        return [
            seg for seg in self.load_segments().values()
            if seg.situation_id == situation_id
        ]
    
    def get_suggestions_for_segment(self, segment_id: str) -> List[Suggestion]:
        """Get all suggestions for a given segment."""
        return [
            sug for sug in self.load_suggestions().values()
            if sug.segment_id == segment_id
        ]
    
    def get_checks_for_suggestion(self, suggestion_id: str) -> List[ComplianceCheck]:
        """Get all compliance checks for a given suggestion."""
        suggestion = self.get_suggestion(suggestion_id)
        if not suggestion:
            return []
        
        checks = self.load_compliance_checks()
        return [
            checks[check_id] for check_id in suggestion.compliance_checks
            if check_id in checks
        ]
    
    def reload_all(self):
        """Reload all configuration from disk."""
        self.load_situations(force_reload=True)
        self.load_segments(force_reload=True)
        self.load_suggestions(force_reload=True)
        self.load_compliance_checks(force_reload=True)


# Global instance for convenient access
_global_loader = None

def get_config_loader(config_dir: Optional[Path] = None) -> ConfigLoader:
    """
    Get the global ConfigLoader instance.
    
    Args:
        config_dir: Path to config directory (only used on first call)
        
    Returns:
        Global ConfigLoader instance
    """
    global _global_loader
    if _global_loader is None:
        _global_loader = ConfigLoader(config_dir)
    return _global_loader


# Convenience functions
def load_situations() -> Dict[str, Situation]:
    """Load all situations using global loader."""
    return get_config_loader().load_situations()


def load_segments() -> Dict[str, Segmentation]:
    """Load all segments using global loader."""
    return get_config_loader().load_segments()


def load_suggestions() -> Dict[str, Suggestion]:
    """Load all suggestions using global loader."""
    return get_config_loader().load_suggestions()


def load_compliance_checks() -> Dict[str, ComplianceCheck]:
    """Load all compliance checks using global loader."""
    return get_config_loader().load_compliance_checks()
