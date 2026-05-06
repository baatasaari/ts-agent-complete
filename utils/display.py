"""
Display Utilities
=================
Modular display functions for Zone 2 demo output.
Configuration-driven visual formatting.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class DisplayConfig:
    """Load and manage display configuration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "display.yaml"
        
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.display = self.config["display"]
        self.emojis = self.config["emojis"]
        self.verbosity = self.config["verbosity"]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value if value is not None else default
    
    def emoji(self, key: str) -> str:
        """Get an emoji by key."""
        return self.emojis.get(key, "")


class DisplayFormatter:
    """Handles all visual formatting for demo output."""
    
    def __init__(self, config: Optional[DisplayConfig] = None):
        self.config = config or DisplayConfig()
    
    def header(self, title: str, char: Optional[str] = None) -> str:
        """Format a header."""
        if char is None:
            char = self.config.get("display.headers.primary_char", "=")
        width = self.config.get("display.headers.width", 70)
        
        lines = [
            "",
            char * width,
            f"  {title}",
            char * width,
            ""
        ]
        return "\n".join(lines)
    
    def section(self, emoji_key: str, title: str) -> str:
        """Format a section header."""
        emoji = self.config.emoji(emoji_key)
        return f"\n{emoji} {title}\n"
    
    def progress_bar(self, ratio: float, length: Optional[int] = None) -> str:
        """Create a progress bar."""
        if length is None:
            length = self.config.get("display.progress_bar.length", 30)
        filled_char = self.config.get("display.progress_bar.filled_char", "█")
        empty_char = self.config.get("display.progress_bar.empty_char", "░")
        
        filled = int(ratio * length)
        return f"[{filled_char * filled}{empty_char * (length - filled)}] {ratio:.0%}"
    
    def confidence_bar(self, confidence: float) -> str:
        """Create a confidence visualization bar."""
        length = self.config.get("display.confidence_bar.length", 20)
        filled_char = self.config.get("display.confidence_bar.filled_char", "█")
        filled = int(confidence * length)
        return filled_char * filled


class GraphDisplay:
    """Display graph state information."""
    
    def __init__(self, formatter: Optional[DisplayFormatter] = None):
        self.formatter = formatter or DisplayFormatter()
    
    def show_state(self, graph) -> str:
        """Display current graph completeness."""
        known = list(graph.known_nodes())
        missing = list(graph.missing_nodes())
        total = len(graph.nodes)
        
        completeness = len(known) / total if total > 0 else 0
        
        lines = []
        lines.append(f"   Graph Completeness: {self.formatter.progress_bar(completeness)}")
        lines.append(f"   Known: {len(known)}/{total} traits")
        
        if missing:
            max_display = self.formatter.config.get("display.graph.max_missing_display", 5)
            missing_ids = [n.char_id for n in missing[:max_display]]
            lines.append(f"   Missing: {', '.join(missing_ids)}")
            if len(missing) > max_display:
                lines.append(f"            ... and {len(missing) - max_display} more")
        lines.append("")
        
        return "\n".join(lines)


class MLPredictionDisplay:
    """Display ML prediction information."""
    
    def __init__(self, formatter: Optional[DisplayFormatter] = None):
        self.formatter = formatter or DisplayFormatter()
    
    def show_prediction(self, hypothesis: Dict, turn: int) -> str:
        """Display ML prediction with SHAP features."""
        lines = []
        
        emoji = self.formatter.config.emoji("ml_prediction")
        lines.append(f"\n   {emoji} ML Prediction (Turn {turn}):")
        lines.append(f"      Segment: {hypothesis.get('top_segment_id', 'N/A')}")
        
        confidence = hypothesis.get('top_confidence', 0)
        pct_format = self.formatter.config.get("display.top_segments.percentage_format", ".1%")
        lines.append(f"      Confidence: {confidence:{pct_format}}")
        
        # Show top segments
        if 'all_scores' in hypothesis:
            scores = hypothesis['all_scores']
            max_display = self.formatter.config.get("display.top_segments.max_display", 3)
            top_n = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_display]
            
            segments_str = " | ".join([f"{seg}: {score:{pct_format}}" for seg, score in top_n])
            lines.append(f"      Top {max_display}: {segments_str}")
        
        # Show SHAP features
        if 'shap_features' in hypothesis and hypothesis['shap_features']:
            lines.append(f"      Key Features (SHAP):")
            max_features = self.formatter.config.get("display.shap_features.max_display", 3)
            decimal_places = self.formatter.config.get("display.shap_features.decimal_places", 3)
            
            for feat in hypothesis['shap_features'][:max_features]:
                lines.append(f"         • {feat['f']}: {feat['v']:.{decimal_places}f}")
        
        lines.append("")
        return "\n".join(lines)
    
    def show_history(self, history: list) -> str:
        """Display prediction evolution."""
        if not history:
            return ""
        
        emoji = self.formatter.config.emoji("evolution")
        lines = [f"\n{emoji} ML Confidence Evolution\n"]
        
        for entry in history:
            turn = entry.get('turn', 0)
            segment = entry.get('top_segment_id', 'N/A')
            confidence = entry.get('top_confidence', 0)
            delta = entry.get('confidence_delta', 0)
            
            bar = self.formatter.confidence_bar(confidence)
            lines.append(f"   Turn {turn:2d}: {segment:15s} {confidence:5.1%} [{bar:20s}] Δ{delta:+.1%}")
        
        lines.append("")
        return "\n".join(lines)


class MetricsDisplay:
    """Display performance metrics."""
    
    def __init__(self, formatter: Optional[DisplayFormatter] = None):
        self.formatter = formatter or DisplayFormatter()
    
    def show_timing(self, duration: float, label: str = "completed") -> str:
        """Show timing information."""
        emoji = self.formatter.config.emoji("timing")
        decimal_places = self.formatter.config.get("display.metrics.time_decimal_places", 2)
        return f"   {emoji}  {label} in {duration:.{decimal_places}f}s"
    
    def show_turn_metrics(self, turn_times: list) -> str:
        """Show turn metrics summary."""
        if not turn_times:
            return ""
        
        lines = []
        avg_turn = sum(turn_times) / len(turn_times)
        decimal_places = self.formatter.config.get("display.metrics.time_decimal_places", 2)
        
        lines.append(f"Turn Metrics:")
        lines.append(f"  Total Turns: {len(turn_times)}")
        lines.append(f"  Avg Turn Time: {avg_turn:.{decimal_places}f}s")
        
        if self.formatter.config.get("display.metrics.show_min_max", True):
            lines.append(f"  Min: {min(turn_times):.{decimal_places}f}s, Max: {max(turn_times):.{decimal_places}f}s")
        
        return "\n".join(lines)


# Convenience function for quick access
def get_display_formatter(config_path: Optional[Path] = None) -> DisplayFormatter:
    """Get a configured display formatter."""
    config = DisplayConfig(config_path)
    return DisplayFormatter(config)
