
from typing import Any, Dict, Optional, Tuple
import simpy

import FJSPSimulation
from enums.AgentType import AgentType
from enums.LocationType import LocationType
from enums.PackagingColor import PackagingColor
from enums.ProductType import ProductType

from .BaseAgent import BaseAgent
from models.Tray import Tray
from gymnasium import spaces
from utils.ObservationSpaces import ObservationSpaces
from utils.ActionSpaces import ActionSpaces
from utils.Logger import get_logger
from constants import CONFIG, LOCATION_POSITIONS
import numpy as np

logger = get_logger()


class AGVAgent(BaseAgent):
    """
    AGV (Automated Guided Vehicle) Agent

    Responsibilities:
    - Navigate grid world
    - Pick up and drop off trays
    - Transport between stations

    SimPy Integration:
    - Movement takes time based on Manhattan distance
    - Uses env.timeout() for movement duration
    """

    def __init__(self, env: simpy.Environment, simulation: FJSPSimulation):
        super().__init__("agv", AgentType.AGV, env)
        self.simulation: FJSPSimulation = simulation

        self.position: Tuple[int, int] = LOCATION_POSITIONS[LocationType.PICKUP]
        self.carrying_tray: Optional[Tray] = None
        self.target_location = None
        self.is_moving: bool = False
        self.movement_process: Optional[simpy.Process] = None

    def get_observation_space(self) -> spaces.Space:
        return ObservationSpaces.agv()

    def get_action_space(self) -> spaces.Space:
        return ActionSpaces.agv()

    def get_observation(self) -> Dict[str, np.ndarray]:
        # Determine tray type (0=none, 1=SMALL, 2=MEDIUM, 3=BIG)
        tray_type = 0
        if self.carrying_tray and self.carrying_tray.tray_type:
            type_map = {'SMALL': 1, 'MEDIUM': 2, 'BIG': 3}
            tray_type = type_map.get(self.carrying_tray.tray_type, 0)

        observation = {
            'position': np.array(self.position, dtype=np.int32),
            'carrying_tray': np.array(1 if self.carrying_tray else 0, dtype=np.int32),
            'tray_product_count': np.array(len(self.carrying_tray.products) if self.carrying_tray else 0, dtype=np.int32),
            'tray_type': np.array(tray_type, dtype=np.int32),  # 0=none, 1=SMALL, 2=MEDIUM, 3=BIG
            'tray_needs_processing': np.array(1 if self.carrying_tray and self.carrying_tray.needs_processing else 0, dtype=np.int32),
            'tray_needs_packaging': np.array(1 if self.carrying_tray and self.carrying_tray.needs_packaging else 0, dtype=np.int32),
            # Ready trays at pickup
            'pickup_ready_trays': np.array(len(self.simulation.pickup_station.ready_trays), dtype=np.int32),
            'small_machine_busy': np.array(1 if self.simulation.small_machine.is_busy else 0, dtype=np.int32),
            'big_machine_busy': np.array(1 if self.simulation.big_machine.is_busy else 0, dtype=np.int32),
            'small_machine_ready': np.array(len(self.simulation.small_machine.ready_trays), dtype=np.int32),
            'big_machine_ready': np.array(len(self.simulation.big_machine.ready_trays), dtype=np.int32),
            'storage_tray_count': np.array(len(self.simulation.storage.trays), dtype=np.int32),
            'action_mask': self.get_action_mask().astype(np.int8)
        }
        return observation
    

    def get_action_mask(self) -> np.ndarray:
        """
        Generates a boolean mask for valid actions based on current state.
        1 = Valid, 0 = Invalid.
        
        Action Space (8):
        0: IDLE
        1: MOVE_TO_PICKUP
        2: MOVE_TO_SMALL_MACHINE
        3: MOVE_TO_BIG_MACHINE
        4: MOVE_TO_STORAGE
        5: MOVE_TO_PACKAGING
        6: PICKUP_TRAY
        7: DROP_TRAY
        """
        # Initialize all as 0 (Invalid)
        mask = np.zeros(8, dtype=np.int32)
        
        # 0. IDLE is always valid (safety fallback)
        mask[0] = 1

        # If AGV is currently moving, it cannot take new actions until arrival
        if self.is_moving:
            return mask  # Returns [1, 0, 0, ...]
        
        current_loc = self._get_current_location()

        # --- MOVEMENT ACTIONS (1-5) ---
        # Map actions to LocationTypes to prevent moving to current location
        # Action index -> LocationType
        move_actions = {
            1: LocationType.PICKUP,
            2: LocationType.SMALL_MACHINE,
            3: LocationType.BIG_MACHINE,
            4: LocationType.STORAGE,
            5: LocationType.PACKAGING
        }
        
        for action_idx, target_loc in move_actions.items():
            # Valid if we are NOT currently at the target location
            if current_loc != target_loc:
                mask[action_idx] = 1

        # --- MANIPULATION ACTIONS (6-7) ---
        
        # 6. PICKUP_TRAY
        # Prereqs: Must NOT be holding a tray, and resources must be available at current loc
        if self.carrying_tray is None and current_loc is not None:
            if current_loc == LocationType.PICKUP:
                # Valid if pickup station has ready trays
                if len(self.simulation.pickup_station.ready_trays) > 0:
                    mask[6] = 1
            
            elif current_loc == LocationType.SMALL_MACHINE:
                # Valid if machine has processed output ready
                if len(self.simulation.small_machine.ready_trays) > 0:
                    mask[6] = 1
                    
            elif current_loc == LocationType.BIG_MACHINE:
                # Valid if machine has processed output ready
                if len(self.simulation.big_machine.ready_trays) > 0:
                    mask[6] = 1
                    
            elif current_loc == LocationType.STORAGE:
                # Valid if storage has any trays
                if len(self.simulation.storage.trays) > 0:
                    mask[6] = 1
            
            # Note: Cannot pickup from PACKAGING (as per _execute_pickup logic)

        # 7. DROP_TRAY
        # Prereqs: Must BE holding a tray, and tray must match destination constraints
        elif self.carrying_tray is not None and current_loc is not None:
            tray = self.carrying_tray
            
            if current_loc == LocationType.PICKUP:
                # Can only drop EMPTY trays at pickup (returning them)
                if not tray.products:
                    mask[7] = 1
            
            elif current_loc == LocationType.SMALL_MACHINE:
                # Needs processing + Compatible Type (SMALL/MEDIUM)
                if tray.needs_processing and tray.tray_type in [ProductType.SMALL.name, ProductType.MEDIUM.name]:
                    mask[7] = 1
            
            elif current_loc == LocationType.BIG_MACHINE:
                # Needs processing + Compatible Type (BIG/MEDIUM)
                if tray.needs_processing and tray.tray_type in [ProductType.BIG.name, ProductType.MEDIUM.name]:
                    mask[7] = 1
            
            elif current_loc == LocationType.PACKAGING:
                # Needs packaging + Processing complete
                if tray.needs_packaging and not tray.needs_processing:
                    mask[7] = 1
            
            elif current_loc == LocationType.STORAGE:
                # Can always drop at storage
                mask[7] = 1

        return mask

    def execute_action(self, action: int) -> Dict[str, Any]:
        """
        Execute AGV action.

        Actions:
            0 = IDLE
            1 = MOVE_TO_PICKUP
            2 = MOVE_TO_SMALL_MACHINE
            3 = MOVE_TO_BIG_MACHINE
            4 = MOVE_TO_STORAGE
            5 = MOVE_TO_PACKAGING
            6 = PICKUP_TRAY
            7 = DROP_TRAY

        Returns result dict for reward calculation.
        """
        result = {
            'action': action,
            'success': False,
            'invalid_action': False,
            'moved': False,
            'distance': 0,
            'pickup_success': False,
            'drop_success': False,
            'delivered_to_packaging': False,
        }

        

        # Can't do anything while moving
        if self.is_moving:
            result['invalid_action'] = True
            return result

        if action == 0:  # IDLE
            result['success'] = True

        elif 1 <= action <= 5:  # MOVE_TO_[LOCATION]
            location_map = {
                1: LocationType.PICKUP,
                2: LocationType.SMALL_MACHINE,
                3: LocationType.BIG_MACHINE,
                4: LocationType.STORAGE,
                5: LocationType.PACKAGING,
            }
            target_location = location_map[action]
            target_pos = LOCATION_POSITIONS[target_location]

            # Calculate Manhattan distance
            distance = abs(self.position[0] - target_pos[0]) + abs(self.position[1] - target_pos[1])

            if distance == 0:
                result['success'] = True
                result['distance'] = 0
            else:
                self.target_position = target_pos
                self.env.process(self._move_process(target_pos))
                logger.agv_move(self.position, target_pos, distance)

                result['success'] = True
                result['moved'] = True
                result['distance'] = distance

        elif action == 6:  # PICKUP_TRAY
            result = self._execute_pickup(result)

        elif action == 7:  # DROP_TRAY
            result = self._execute_drop(result)

        else:
            result['invalid_action'] = True

        return result

    def _execute_pickup(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tray pickup at current location."""

        if self.carrying_tray is not None:
            result['invalid_action'] = True
            return result

        current_location = self._get_current_location()

        if current_location is None:
            result['invalid_action'] = True
            return result

        tray = None

        if current_location == LocationType.PICKUP:
            tray = self.simulation.pickup_station.get_ready_tray()

        elif current_location == LocationType.SMALL_MACHINE:
            tray = self.simulation.small_machine.get_ready_tray()

        elif current_location == LocationType.BIG_MACHINE:
            tray = self.simulation.big_machine.get_ready_tray()

        elif current_location == LocationType.STORAGE:
            tray = self.simulation.storage.get_tray()

        elif current_location == LocationType.PACKAGING:
            result['invalid_action'] = True
            return result

        if tray is not None:
            self.carrying_tray = tray
            result['success'] = True
            result['pickup_success'] = True
            logger.tray_pickup(tray.id, current_location.name)
        else:
            result['invalid_action'] = True

        return result

    def _execute_drop(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tray drop at current location."""

        if self.carrying_tray is None:
            result['invalid_action'] = True
            return result

        current_location = self._get_current_location()

        if current_location is None:
            result['invalid_action'] = True
            return result

        drop_success = False

        if current_location == LocationType.PICKUP:
            if not self.carrying_tray.products:
                self.carrying_tray.change_location_to(LocationType.PICKUP)
                self.simulation.pickup_station.add_empty_tray(self.carrying_tray)
                drop_success = True
            else:
                result['invalid_action'] = True
                return result

        elif current_location == LocationType.SMALL_MACHINE:
            if self.carrying_tray.needs_processing:
                tray_type = self.carrying_tray.tray_type
                if tray_type in [ProductType.SMALL.name, ProductType.MEDIUM.name]:
                    self.carrying_tray.change_location_to(LocationType.SMALL_MACHINE)
                    self.simulation.small_machine.add_tray(self.carrying_tray)
                    drop_success = True
                else:
                    result['invalid_action'] = True
                    return result
            else:
                result['invalid_action'] = True
                return result

        elif current_location == LocationType.BIG_MACHINE:
            if self.carrying_tray.needs_processing:
                tray_type = self.carrying_tray.tray_type
                if tray_type in [ProductType.BIG.name, ProductType.MEDIUM.name]:
                    self.carrying_tray.change_location_to(LocationType.BIG_MACHINE)
                    self.simulation.big_machine.add_tray(self.carrying_tray)
                    drop_success = True
                else:
                    result['invalid_action'] = True
                    return result
            else:
                result['invalid_action'] = True
                return result

        elif current_location == LocationType.STORAGE:
            self.carrying_tray.change_location_to(LocationType.STORAGE)
            self.simulation.storage.add_tray(self.carrying_tray)
            drop_success = True

        elif current_location == LocationType.PACKAGING:
            if self.carrying_tray.needs_packaging and (not self.carrying_tray.needs_processing):
                self.carrying_tray.change_location_to(LocationType.PACKAGING)
                self.simulation.add_tray_to_packaging(self.carrying_tray)
                drop_success = True
                result['delivered_to_packaging'] = True
            else:
                result['invalid_action'] = True
                return result

        if drop_success:
            logger.tray_drop(self.carrying_tray.id, current_location.name)
            self.carrying_tray = None
            result['success'] = True
            result['drop_success'] = True

        return result

    def _calculate_current_location(self) -> int:
        """Determine which location AGV is at based on grid position."""
        for loc, pos in LOCATION_POSITIONS.items():
            if self.position == pos:
                return loc
        return None

    def _simpy_movement_process(self, target_pos: Tuple[int, int]):
        """SimPy process for AGV movement."""
        distance = abs(self.position[0] - target_pos[0]) + abs(self.position[1] - target_pos[1])
        travel_time = distance / CONFIG['agv_speed']

        self.is_moving = True
        yield self.env.timeout(travel_time)
        self.position = target_pos
        self.is_moving = False

    def _move_process(self, target_pos: Tuple[int, int]):
        """SimPy process for AGV movement."""
        distance = abs(self.position[0] - target_pos[0]) + abs(self.position[1] - target_pos[1])
        travel_time = distance / CONFIG['agv_speed']

        self.is_moving = True
        yield self.env.timeout(travel_time)
        self.position = target_pos
        self.is_moving = False
        self.target_position = None

    def _get_current_location(self) -> Optional[LocationType]:
        """Determine which location AGV is at based on grid position."""
        for location, pos in LOCATION_POSITIONS.items():
            if self.position == pos:
                return location
        return None  # In transit
