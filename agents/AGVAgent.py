
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
from constants import CONFIG, LOCATION_POSITIONS
import numpy as np


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
        self.simulation : FJSPSimulation = simulation
        
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
        observation = {
            'position': np.array(self.position, dtype=np.int32),
            'carrying_tray': np.array(1 if self.carrying_tray else 0, dtype=np.int32),
            'tray_product_count': np.array(len(self.carrying_tray.products) if self.carrying_tray else 0, dtype=np.int32),
            'tray_needs_processing': np.array(1 if self.carrying_tray and self.carrying_tray.needs_machine_processing else 0, dtype=np.int32),
            'tray_needs_packaging': np.array(1 if self.carrying_tray and self.carrying_tray.needs_packaging else 0, dtype=np.int32),
            
            # Use correct attribute names from FJSPSimulation
            'pickup_queue': np.array(len(self.simulation.pickup_station.order_queue), dtype=np.int32),
            'small_machine_busy': np.array(1 if self.simulation.small_machine.is_busy else 0, dtype=np.int32),
            'big_machine_busy': np.array(1 if self.simulation.big_machine.is_busy else 0, dtype=np.int32),
            'storage_tray_count': np.array(len(self.simulation.storage.trays), dtype=np.int32),
            
            # Packaging queues - get from each station
            'packaging_queues': np.array([
                len(self.simulation.packaging_stations['packaging_blue_1'].product_queue),
                len(self.simulation.packaging_stations['packaging_blue_2'].product_queue),
                len(self.simulation.packaging_stations['packaging_red'].product_queue),
                len(self.simulation.packaging_stations['packaging_green'].product_queue),
            ], dtype=np.int32),
        }
        return observation


    
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
                # Already at target
                result['success'] = True
                result['distance'] = 0
            else:
                # Start movement (SimPy process)
                self.target_position = target_pos
                self.env.process(self._move_process(target_pos))
                
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
        
        # Can't pickup if already carrying
        if self.carrying_tray is not None:
            result['invalid_action'] = True
            return result
        
        current_location = self._get_current_location()
        
        if current_location is None:
            # Not at any station
            result['invalid_action'] = True
            return result
        
        # Try to get tray from current location
        tray = None
        
        if current_location == LocationType.PICKUP:
            tray = self.simulation.pickup_station_agent.get_ready_tray()
            
        elif current_location == LocationType.SMALL_MACHINE:
            tray = self.simulation.small_machine.get_ready_tray()
            
        elif current_location == LocationType.BIG_MACHINE:
            tray = self.simulation.big_machine.get_ready_tray()
            
        elif current_location == LocationType.STORAGE:
            tray = self.simulation.storage.get_tray()
            
        elif current_location == LocationType.PACKAGING:
            # Typically don't pickup from packaging (products leave system)
            result['invalid_action'] = True
            return result
        
        if tray is not None:
            self.carrying_tray = tray
            result['success'] = True
            result['pickup_success'] = True
        else:
            # No tray available at this location
            result['invalid_action'] = True
        
        return result
    

    def _execute_drop(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tray drop at current location."""
        
        # Can't drop if not carrying
        if self.carrying_tray is None:
            result['invalid_action'] = True
            return result
        
        current_location = self._get_current_location()
        
        if current_location is None:
            # Not at any station
            result['invalid_action'] = True
            return result
        
        drop_success = False
        
        if current_location == LocationType.PICKUP:
            # Drop empty tray back at pickup
            if not self.carrying_tray.products:
                self.simulation.pickup_station_agent.add_empty_tray(self.carrying_tray)
                drop_success = True
            else:
                result['invalid_action'] = True
                return result
                
        elif current_location == LocationType.SMALL_MACHINE:
            # Drop tray for processing (must need processing and be SMALL or MEDIUM type)
            if self.carrying_tray.needs_processing:
                tray_type = self.carrying_tray.tray_type
                if tray_type in [ProductType.SMALL, ProductType.MEDIUM]:
                    self.simulation.small_machine_agent.add_tray(self.carrying_tray)
                    drop_success = True
                else:
                    result['invalid_action'] = True
                    return result
            else:
                result['invalid_action'] = True
                return result
                
        elif current_location == LocationType.BIG_MACHINE:
            # Drop tray for processing (must need processing and be BIG or MEDIUM type)
            if self.carrying_tray.needs_processing:
                tray_type = self.carrying_tray.tray_type
                if tray_type in [ProductType.BIG, ProductType.MEDIUM]:
                    self.simulation.big_machine_agent.add_tray(self.carrying_tray)
                    drop_success = True
                else:
                    result['invalid_action'] = True
                    return result
            else:
                result['invalid_action'] = True
                return result
                
        elif current_location == LocationType.STORAGE:
            # Can always drop at storage (buffer)
            self.simulation.storage_area.add_tray(self.carrying_tray)
            drop_success = True
            
        elif current_location == LocationType.PACKAGING:
            # Drop processed tray for packaging
            if self.carrying_tray.needs_packaging:
                self.simulation.packaging_agent.add_tray(self.carrying_tray)
                drop_success = True
                result['delivered_to_packaging'] = True
            else:
                result['invalid_action'] = True
                return result
        
        if drop_success:
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
        """
        SimPy process for AGV movement.
        Yields timeout based on distance.
        """
        distance = abs(self.position[0] - target_pos[0]) + abs(self.position[1] - target_pos[1])
        travel_time = distance / CONFIG['agv_speed']
        
        self.is_moving = True
        yield self.env.timeout(travel_time)
        self.position = target_pos
        self.is_moving = False

    def _move_process(self, target_pos: Tuple[int, int]):
        """
        SimPy process for AGV movement.
        Takes time based on Manhattan distance.
        """
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