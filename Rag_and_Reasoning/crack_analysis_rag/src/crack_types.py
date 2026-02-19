"""
Crack type definitions for structural analysis
"""
from enum import Enum
from typing import Dict, List


class CrackType(Enum):
    """Enumeration of structural crack types"""
    DIAGONAL = "diagonal"
    STEP = "step"
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    X_SHAPED = "x"
    
    def __str__(self):
        return self.value


class CrackCharacteristics:
    """Contains characteristics and common causes for each crack type"""
    
    CRACK_INFO = {
        CrackType.DIAGONAL: {
            "description": "Diagonal cracks that appear at angles across the structure",
            "common_locations": ["walls", "beams", "slabs"],
            "typical_causes": [
                "Differential settlement",
                "Thermal expansion/contraction",
                "Structural movement",
                "Foundation issues",
                "Seismic activity"
            ],
            "analysis_keywords": ["shear stress", "foundation settlement", "thermal movement", "structural instability"]
        },
        
        CrackType.STEP: {
            "description": "Step-like cracks that follow mortar joints in masonry",
            "common_locations": ["brick walls", "block walls", "masonry structures"],
            "typical_causes": [
                "Foundation settlement",
                "Thermal expansion",
                "Moisture movement",
                "Structural overloading",
                "Poor construction quality"
            ],
            "analysis_keywords": ["masonry settlement", "thermal stress", "moisture expansion", "structural loading"]
        },
        
        CrackType.VERTICAL: {
            "description": "Vertical cracks running up and down the structure",
            "common_locations": ["walls", "columns", "vertical elements"],
            "typical_causes": [
                "Thermal expansion/contraction",
                "Drying shrinkage",
                "Overloading",
                "Foundation movement",
                "Construction joints"
            ],
            "analysis_keywords": ["thermal stress", "shrinkage", "vertical loading", "construction defects"]
        },
        
        CrackType.HORIZONTAL: {
            "description": "Horizontal cracks running along the length of structure",
            "common_locations": ["beams", "walls", "horizontal elements"],
            "typical_causes": [
                "Flexural stress",
                "Overloading",
                "Reinforcement corrosion",
                "Temperature effects",
                "Structural deflection"
            ],
            "analysis_keywords": ["bending stress", "structural overload", "reinforcement failure", "deflection"]
        },
        
        CrackType.X_SHAPED: {
            "description": "X-shaped or cross-pattern cracks",
            "common_locations": ["shear walls", "panels", "structural elements"],
            "typical_causes": [
                "Shear failure",
                "Seismic activity",
                "Lateral loads",
                "Structural instability",
                "Foundation problems"
            ],
            "analysis_keywords": ["shear failure", "seismic damage", "lateral forces", "structural collapse"]
        }
    }
    
    @classmethod
    def get_crack_info(cls, crack_type: CrackType) -> Dict:
        """Get detailed information about a specific crack type"""
        return cls.CRACK_INFO.get(crack_type, {})
    
    @classmethod
    def get_analysis_keywords(cls, crack_type: CrackType) -> List[str]:
        """Get keywords for RAG document retrieval"""
        info = cls.get_crack_info(crack_type)
        return info.get("analysis_keywords", [])
    
    @classmethod
    def get_typical_causes(cls, crack_type: CrackType) -> List[str]:
        """Get typical causes for a crack type"""
        info = cls.get_crack_info(crack_type)
        return info.get("typical_causes", [])
