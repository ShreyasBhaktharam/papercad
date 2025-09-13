import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from .primitives import Point, LineSegment, Arc

@dataclass
class Constraint:
    """Represents a geometric constraint with priority and confidence"""
    type: str  # 'perpendicular', 'parallel', 'tangent', etc.
    elements: Tuple  # Indices of geometric elements involved
    priority: float  # Higher values = higher priority
    confidence: float  # 0.0 to 1.0, from AI detection confidence
    source: str  # 'ai_detection', 'geometric_inference', 'user_input'

class ConflictResolver:
    """Resolves conflicts between competing geometric constraints"""
    
    def __init__(self):
        self.priority_weights = {
            'user_input': 1.0,      # Highest priority
            'ai_detection': 0.8,    # High confidence AI detections
            'geometric_inference': 0.6  # Inferred from geometry analysis
        }
        
        self.constraint_priorities = {
            'perpendicular': 0.9,   # Very important for architectural drawings
            'parallel': 0.8,       # Important for structural consistency
            'equal_length': 0.7,   # Important for symmetry
            'tangent': 0.6,        # Important for precise connections
            'collinear': 0.5,      # Less critical
            'concentric': 0.4      # Least critical for basic functionality
        }
    
    def resolve_conflicts(self, 
                         constraints_dict: Dict[str, List[Tuple]], 
                         ai_confidences: Dict = None) -> Dict[str, List[Tuple]]:
        """
        Resolve conflicts between competing constraints
        
        Args:
            constraints_dict: Dictionary of constraint type -> list of element pairs
            ai_confidences: Optional confidence scores from AI detections
            
        Returns:
            Filtered dictionary with conflicts resolved
        """
        # Convert to Constraint objects with priorities
        all_constraints = self._create_constraint_objects(constraints_dict, ai_confidences)
        
        # Find conflicts
        conflicts = self._detect_conflicts(all_constraints)
        
        # Resolve conflicts by priority
        resolved_constraints = self._resolve_by_priority(all_constraints, conflicts)
        
        # Convert back to original format
        return self._constraints_to_dict(resolved_constraints)
    
    def _create_constraint_objects(self, 
                                  constraints_dict: Dict[str, List[Tuple]], 
                                  ai_confidences: Dict = None) -> List[Constraint]:
        """Convert constraint dictionary to Constraint objects"""
        constraints = []
        
        for constraint_type, element_pairs in constraints_dict.items():
            for elements in element_pairs:
                # Calculate priority based on type and source
                base_priority = self.constraint_priorities.get(constraint_type, 0.5)
                
                # Get confidence from AI if available
                confidence = 0.8  # Default confidence
                if ai_confidences and constraint_type in ai_confidences:
                    confidence = ai_confidences[constraint_type]
                
                # Determine source (simplified - assumes geometric inference)
                source = 'geometric_inference'
                if confidence > 0.9:
                    source = 'ai_detection'
                
                priority = base_priority * self.priority_weights[source] * confidence
                
                constraints.append(Constraint(
                    type=constraint_type,
                    elements=elements,
                    priority=priority,
                    confidence=confidence,
                    source=source
                ))
        
        return constraints
    
    def _detect_conflicts(self, constraints: List[Constraint]) -> List[Set[int]]:
        """Detect conflicting constraints"""
        conflicts = []
        
        for i, constraint1 in enumerate(constraints):
            for j, constraint2 in enumerate(constraints[i+1:], start=i+1):
                if self._are_conflicting(constraint1, constraint2):
                    # Find existing conflict group or create new one
                    conflict_group = None
                    for group in conflicts:
                        if i in group or j in group:
                            conflict_group = group
                            break
                    
                    if conflict_group:
                        conflict_group.update([i, j])
                    else:
                        conflicts.append({i, j})
        
        return conflicts
    
    def _are_conflicting(self, constraint1: Constraint, constraint2: Constraint) -> bool:
        """Check if two constraints conflict with each other"""
        # Same elements can't have conflicting geometric relationships
        if constraint1.elements == constraint2.elements:
            return self._constraint_types_conflict(constraint1.type, constraint2.type)
        
        # Check for transitive conflicts (A parallel B, B parallel C, A perpendicular C)
        if len(constraint1.elements) == 2 and len(constraint2.elements) == 2:
            shared_elements = set(constraint1.elements) & set(constraint2.elements)
            if len(shared_elements) == 1:
                return self._check_transitive_conflict(constraint1, constraint2)
        
        return False
    
    def _constraint_types_conflict(self, type1: str, type2: str) -> bool:
        """Check if two constraint types are incompatible"""
        conflicts = {
            ('perpendicular', 'parallel'),
            ('perpendicular', 'collinear'),
            ('parallel', 'perpendicular'),
            ('collinear', 'perpendicular')
        }
        return (type1, type2) in conflicts or (type2, type1) in conflicts
    
    def _check_transitive_conflict(self, constraint1: Constraint, constraint2: Constraint) -> bool:
        """Check for transitive conflicts (e.g., A||B, B⊥C implies A⊥C but we have A||C)"""
        # This is a simplified check - full implementation would require graph analysis
        if constraint1.type == 'parallel' and constraint2.type == 'perpendicular':
            return True
        if constraint1.type == 'perpendicular' and constraint2.type == 'parallel':
            return True
        return False
    
    def _resolve_by_priority(self, 
                            constraints: List[Constraint], 
                            conflicts: List[Set[int]]) -> List[Constraint]:
        """Resolve conflicts by keeping highest priority constraints"""
        to_remove = set()
        
        for conflict_group in conflicts:
            # Sort by priority (descending)
            sorted_constraints = sorted(
                [(i, constraints[i]) for i in conflict_group],
                key=lambda x: x[1].priority,
                reverse=True
            )
            
            # Keep only the highest priority constraint from each conflict group
            for i, _ in sorted_constraints[1:]:
                to_remove.add(i)
        
        # Return constraints excluding removed ones
        return [constraint for i, constraint in enumerate(constraints) if i not in to_remove]
    
    def _constraints_to_dict(self, constraints: List[Constraint]) -> Dict[str, List[Tuple]]:
        """Convert Constraint objects back to dictionary format"""
        result = {}
        
        for constraint in constraints:
            if constraint.type not in result:
                result[constraint.type] = []
            result[constraint.type].append(constraint.elements)
        
        return result
    
    def get_conflict_report(self, constraints_dict: Dict[str, List[Tuple]]) -> Dict:
        """Generate a report of detected conflicts and resolutions"""
        all_constraints = self._create_constraint_objects(constraints_dict)
        conflicts = self._detect_conflicts(all_constraints)
        
        return {
            'total_constraints': len(all_constraints),
            'conflicts_detected': len(conflicts),
            'conflict_groups': [list(group) for group in conflicts],
            'resolution_strategy': 'priority_based',
            'priority_weights': self.priority_weights
        }
