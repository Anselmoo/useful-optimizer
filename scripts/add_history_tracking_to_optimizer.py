#!/usr/bin/env python3
"""Helper script to add history tracking to optimizer files.

This script provides a template and instructions for adding history tracking
to optimizers that don't have it yet.

Usage:
    python scripts/add_history_tracking_to_optimizer.py <optimizer_file>
"""

import sys
from pathlib import Path

TEMPLATE = '''
# Step 1: Add track_history parameter to __init__
# Add this parameter to the __init__ method signature:
        track_history: bool = False,

# Step 2: Pass track_history to super().__init__
# Add this to the super().__init__() call:
            track_history=track_history,

# Step 3: Add history recording in search() method
# At the START of each iteration loop, add:
            # Track history if enabled
            if self.track_history:
                self.history["best_fitness"].append(float(best_fitness))
                self.history["best_solution"].append(best_solution.copy())

# Step 4: Add final history recording
# At the END of search(), before return, add:
        # Track final state
        if self.track_history:
            self.history["best_fitness"].append(float(best_fitness))
            self.history["best_solution"].append(best_solution.copy())

# Example: See opt/swarm_intelligence/particle_swarm.py
# Example: See opt/swarm_intelligence/ant_colony.py
# Example: See opt/swarm_intelligence/firefly_algorithm.py
'''

def main():
    """Print instructions for adding history tracking."""
    if len(sys.argv) > 1:
        optimizer_file = Path(sys.argv[1])
        if not optimizer_file.exists():
            print(f"Error: File {optimizer_file} not found")
            sys.exit(1)
        print(f"Adding history tracking to {optimizer_file}")
        print(TEMPLATE)
    else:
        print("Usage: python scripts/add_history_tracking_to_optimizer.py <optimizer_file>")
        print(TEMPLATE)

if __name__ == "__main__":
    main()
