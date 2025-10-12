import os
import importlib.util

def load_module(module_name, file_name):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join('/home/miles/wsl-code/bu/cs664/assignment5/4by4', file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load modules
qtable_module = load_module('q_table', '4q_table.py')
QTable = qtable_module.QTable

# Test the Q-table directly
q_table = QTable(alpha=0.1, gamma=0.9, epsilon=0.2)

# Create a test state
state = {
    'can_win_this_turn': False,
    'must_block_this_turn': False,
    'qty_middle_zone_available': 3,
    'qty_middle_zone_owned': 1,
    'qty_corners_available': 2,
    'qty_edge_mids_available': 4,
    'total_pieces_placed': 2
}

action = 'take_center'

print("Testing Q-table update...")
print(f"Initial Q-table size: {q_table.get_table_size()}")
print(f"Initial Q-value for state-action: {q_table.get_q_value(state, action)}")

# Test update_q_value directly
q_table.update_q_value(state, action, reward=0.0, next_state=None, next_valid_actions=None)

print(f"After update Q-table size: {q_table.get_table_size()}")
print(f"Q-value after update: {q_table.get_q_value(state, action)}")

if q_table.table:
    print(f"\nTable contents:")
    for key, val in q_table.table.items():
        print(f"  {key}: {val}")
else:
    print("\nTable is still empty!")
