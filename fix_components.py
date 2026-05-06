"""Quick script to fix the attribute access issues in both components."""

import re

# Fix decision_spine.py
with open("ts_agent/visualiser/components/decision_spine.py", "r") as f:
    spine_content = f.read()

# Replace .get() calls with direct attribute access for dataclass fields
spine_content = spine_content.replace(
    'len([t for t in record.conversation if t.get("value") is not None])',
    'len(record.conversation)'
)
spine_content = spine_content.replace(
    'len([t for t in record.conversation if t.get("source") == "CONSUMER_INPUT"])',
    'len([t for t in record.conversation if t.source == "CONSUMER_INPUT"])'
)
spine_content = spine_content.replace(
    'top_pred = record.prediction_chain[-1] if record.prediction_chain else {}',
    'top_pred = record.prediction_chain[-1] if record.prediction_chain else None'
)
spine_content = spine_content.replace(
    'confidence = top_pred.get("confidence", 0.0)',
    'confidence = top_pred.top_confidence if top_pred else 0.0'
)
spine_content = spine_content.replace(
    'top_seg = top_pred.get("segment_id", "UNKNOWN")',
    'top_seg = top_pred.top_segment_id if top_pred else "UNKNOWN"'
)
spine_content = spine_content.replace(
    'conf = top_pred.get("confidence", 0.0)',
    'conf = top_pred.top_confidence if top_pred else 0.0'
)

with open("ts_agent/visualiser/components/decision_spine.py", "w") as f:
    f.write(spine_content)

print("✓ Fixed decision_spine.py")

# Fix decision_justification.py
with open("ts_agent/visualiser/components/decision_justification.py", "r") as f:
    just_content = f.read()

# Replace .get() calls for PredictionSnapshot
just_content = just_content.replace(
    'conf = top_pred.get("confidence", 0.0)',
    'conf = top_pred.top_confidence if top_pred else 0.0'
)
just_content = just_content.replace(
    'top_pred = record.prediction_chain[-1] if record.prediction_chain else {}',
    'top_pred = record.prediction_chain[-1] if record.prediction_chain else None'
)

with open("ts_agent/visualiser/components/decision_justification.py", "w") as f:
    f.write(just_content)

print("✓ Fixed decision_justification.py")
print("\nBoth files fixed successfully!")
