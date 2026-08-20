
"""
DCIFM model notes
-----------------
The Streamlit app contains the runnable training pipeline so the project
remains easy to demonstrate. This module documents the intended modelling
logic for research reproducibility.

Important:
- *_ground_truth columns are evaluation-only and must never be model inputs.
- Current and emerging identity are separate prediction targets.
- A transition is reported only when the two predicted identities differ.
- The synthetic dataset is a controlled research dataset, not real consumer data.
"""

from dataclasses import dataclass

@dataclass
class DCIFMDesign:
    current_identity_target: str = "identity_orientation_ground_truth"
    emerging_identity_target: str = "emerging_identity_ground_truth"
    future_category_target: str = "future_need_category_ground_truth"

    def pipeline(self):
        return [
            "Cross-source unified consumer record",
            "Behavioral feature engineering",
            "Current identity prediction",
            "Emerging identity prediction",
            "Current vs emerging comparison",
            "Identity transition strength",
            "Future-need category mapping",
            "Marketing interpretation",
            "Optional visual-context fusion",
        ]
