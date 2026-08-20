"""
ccfv3_aligner.utils.config
Default constants, color maps, and Allen CCFv3 ontology mappings.
"""

# Default target 13 core nuclei + Cortex
BASE_REGION_COLORS = {
    "RSPd": "#ff1a71", "CA3": "#860d4c", "CA1": "#53082f",
    "DG": "#16f2f2", "GPe": "#b199ff", "CP": "#8d7acc",
    "LSc": "#3283fe", "RT": "#ff6600", "TRS": "#7609b1",
    "MH": "#faa307", "LH": "#c68105", "ATN": "#1460ff",
    "RE": "#1340ff", "PF": "#0a4093", "CM": "#08306d",
    "Cortex": "#ffff00"  # Yellow outline
}

TARGET_ACRONYMS = [
    'TRS', 'LSc', 'CP', 'GPe', 'RSPd', 'CA3', 'CA1', 
    'DG', 'RT', 'MH', 'LH', 'ATN', 'RE', 'PF', 'CM', 'Cortex'
]

# CCFv3 Reference Volume Shape & Midline
DEFAULT_MIDLINE_CCF = 228.0
DEFAULT_BREGMA_AP_INDEX = 216
DEFAULT_RESOLUTION_UM = 25.0
