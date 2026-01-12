"""
Centralized Logging for MARL-FJSP Simulation
=============================================

Provides structured, configurable logging for:
- Agent actions and results (per-step granularity)
- Product/tray movements with order context
- Order completions
- Simulation state changes

Logs are written to both console and a timestamped file (logs/log_YYYYMMDD_HHMMSS.txt)
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class LogLevel(Enum):
    """Log levels for simulation events."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    ACTION = 15  # Custom level between DEBUG and INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR


# Register custom ACTION level
logging.addLevelName(LogLevel.ACTION.value, 'ACTION')


class SimulationLogger:
    """Centralized logger for MARL-FJSP simulation."""

    _instance: Optional['SimulationLogger'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.enabled = False  # Disabled by default
        self.log_level = LogLevel.INFO
        self.log_file_path: Optional[str] = None
        self._file_handler: Optional[logging.FileHandler] = None

        # Setup logger
        self.logger = logging.getLogger('FJSP')
        self.logger.setLevel(logging.DEBUG)  # Allow all levels, filter at handler

        # Console handler with formatting
        self._console_handler = logging.StreamHandler()
        self._console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '[T=%(sim_time)-7s] %(message)s',
            defaults={'sim_time': '0'}
        )
        self._console_handler.setFormatter(formatter)

        # Remove existing handlers to avoid duplicates
        self.logger.handlers = []
        self.logger.addHandler(self._console_handler)

        self._sim_time = 0.0
        self._current_step = 0

    def setup_file_logging(self, log_dir: str = "logs") -> str:
        """
        Setup file logging with a timestamped filename.

        Args:
            log_dir: Directory to store log files (default: "logs")

        Returns:
            Path to the created log file
        """
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(log_dir, f"log_{timestamp}.txt")

        # Remove existing file handler if any
        if self._file_handler:
            self.logger.removeHandler(self._file_handler)
            self._file_handler.close()

        # Create file handler
        self._file_handler = logging.FileHandler(self.log_file_path, mode='w')
        self._file_handler.setLevel(logging.DEBUG)

        # Use same format as console but with timestamp
        file_formatter = logging.Formatter(
            '[T=%(sim_time)-7s] %(message)s',
            defaults={'sim_time': '0'}
        )
        self._file_handler.setFormatter(file_formatter)

        self.logger.addHandler(self._file_handler)

        return self.log_file_path

    def close_file_logging(self):
        """Close the file handler and flush all logs."""
        if self._file_handler:
            self._file_handler.flush()
            self._file_handler.close()
            self.logger.removeHandler(self._file_handler)
            self._file_handler = None

    def set_level(self, level: LogLevel):
        """Set the minimum log level."""
        self.log_level = level
        self.logger.setLevel(level.value)

    def set_sim_time(self, time: float):
        """Update current simulation time for log messages."""
        self._sim_time = time

    def set_step(self, step: int):
        """Update current step number."""
        self._current_step = step

    def disable(self):
        """Disable all logging."""
        self.enabled = False

    def enable(self):
        """Enable logging."""
        self.enabled = True

    def _log(self, level: int, msg: str):
        """Internal log method."""
        if not self.enabled:
            return
        self.logger.log(level, msg, extra={'sim_time': f'{self._sim_time:.1f}'})

    # === Step-Level Logging ===

    def step_begin(self, step: int, sim_time: float):
        """Log the beginning of a simulation step."""
        self._current_step = step
        self._sim_time = sim_time
        self._log(logging.DEBUG, f"{'='*60}")
        self._log(logging.DEBUG, f"STEP {step}")
        self._log(logging.DEBUG, f"{'='*60}")

    def step_end(self, step: int, rewards: Dict[str, float], orders_completed: int, products_packaged: int):
        """Log the end of a simulation step with summary."""
        total_reward = sum(rewards.values())
        self._log(logging.DEBUG, f"  Step {step} Summary: reward={total_reward:.2f}, orders={orders_completed}, products={products_packaged}")

    # === Agent Action Logging (Detailed) ===

    def agent_action_detailed(
        self,
        agent_id: str,
        action: int,
        action_name: str,
        result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Log detailed agent action with full context.

        Args:
            agent_id: The agent identifier
            action: Action number
            action_name: Human-readable action name
            result: Action result dictionary
            context: Optional context (order_id, product_id, tray_id, etc.)
        """
        success = result.get('success', False)
        status = "OK" if success else "FAIL"

        # Build context string
        ctx_parts = []
        if context:
            if 'order_id' in context:
                ctx_parts.append(f"order={context['order_id']}")
            if 'product_id' in context:
                ctx_parts.append(f"prod={context['product_id']}")
            if 'tray_id' in context:
                ctx_parts.append(f"tray={context['tray_id']}")
            if 'position' in context:
                ctx_parts.append(f"pos={context['position']}")
            if 'target' in context:
                ctx_parts.append(f"target={context['target']}")

        ctx_str = f" [{', '.join(ctx_parts)}]" if ctx_parts else ""

        # Build result details
        details = []
        for key, val in result.items():
            if key in ('action', 'success'):
                continue
            if val is True:
                details.append(key)
            elif val and val != False:
                details.append(f"{key}={val}")

        detail_str = f" -> {', '.join(details)}" if details else ""

        msg = f"  {agent_id:20s} | act={action} {action_name:18s} | {status}{ctx_str}{detail_str}"
        self._log(LogLevel.ACTION.value, msg)

    def all_agent_actions(
        self,
        actions: Dict[str, int],
        action_names: Dict[str, str],
        results: Dict[str, Dict[str, Any]],
        contexts: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """Log all agent actions for a step in a structured way."""
        self._log(logging.DEBUG, "  ACTIONS:")
        for agent_id, action in actions.items():
            action_name = action_names.get(agent_id, f"ACTION_{action}")
            result = results.get(agent_id, {})
            context = contexts.get(agent_id, {}) if contexts else {}
            self.agent_action_detailed(agent_id, action, action_name, result, context)

    # === Product/Tray Movement (with order context) ===

    def product_loaded(self, order_id: int, product_id: int, tray_id: int):
        """Log when a product is loaded onto a tray."""
        self._log(logging.INFO, f"  PRODUCT LOAD    | Order {order_id} -> Product {product_id} -> Tray {tray_id}")

    def tray_ready(self, tray_id: int, location: str, product_count: int):
        """Log when a tray is ready for pickup."""
        self._log(logging.INFO, f"  TRAY READY      | Tray {tray_id} at {location} ({product_count} products)")

    def tray_pickup(self, tray_id: int, location: str, order_id: Optional[int] = None, product_ids: Optional[List[int]] = None):
        """Log when AGV picks up a tray with full context."""
        ctx = ""
        if order_id is not None:
            ctx = f" [order={order_id}"
            if product_ids:
                ctx += f", products={product_ids}"
            ctx += "]"
        self._log(logging.INFO, f"  AGV PICKUP      | Tray {tray_id} from {location}{ctx}")

    def tray_drop(self, tray_id: int, location: str, order_id: Optional[int] = None, needs_processing: bool = False, needs_packaging: bool = False):
        """Log when AGV drops a tray with state context."""
        ctx_parts = []
        if order_id is not None:
            ctx_parts.append(f"order={order_id}")
        if needs_processing:
            ctx_parts.append("needs_processing")
        if needs_packaging:
            ctx_parts.append("needs_packaging")
        ctx = f" [{', '.join(ctx_parts)}]" if ctx_parts else ""
        self._log(logging.INFO, f"  AGV DROP        | Tray {tray_id} at {location}{ctx}")

    def agv_move(self, from_pos: tuple, to_pos: tuple, distance: int):
        """Log AGV movement."""
        self._log(logging.DEBUG, f"  AGV MOVE        | {from_pos} -> {to_pos} (dist={distance})")

    def agv_state(self, position: tuple, carrying: bool, tray_id: Optional[int] = None, is_moving: bool = False):
        """Log AGV current state."""
        state = f"pos={position}"
        if carrying and tray_id is not None:
            state += f", carrying=Tray {tray_id}"
        elif carrying:
            state += ", carrying=True"
        else:
            state += ", carrying=False"
        if is_moving:
            state += ", MOVING"
        self._log(logging.DEBUG, f"  AGV STATE       | {state}")

    # === Processing (with order/product context) ===

    def processing_start(self, machine: str, tray_id: int, order_id: Optional[int] = None, product_count: int = 0):
        """Log when processing starts."""
        ctx = ""
        if order_id is not None:
            ctx = f" [order={order_id}, {product_count} products]"
        self._log(logging.INFO, f"  MACHINE START   | {machine} processing Tray {tray_id}{ctx}")

    def processing_complete(self, machine: str, tray_id: int, order_id: Optional[int] = None):
        """Log when processing completes."""
        ctx = f" [order={order_id}]" if order_id is not None else ""
        self._log(logging.INFO, f"  MACHINE DONE    | {machine} finished Tray {tray_id}{ctx}")

    def packaging_start(self, station: str, product_id: int, order_id: Optional[int] = None):
        """Log when packaging starts."""
        ctx = f" [order={order_id}]" if order_id is not None else ""
        self._log(logging.INFO, f"  PACKAGING START | {station} -> Product {product_id}{ctx}")

    def packaging_complete(self, station: str, product_id: int, order_id: Optional[int] = None):
        """Log when packaging completes."""
        ctx = f" [order={order_id}]" if order_id is not None else ""
        self._log(logging.INFO, f"  PACKAGING DONE  | {station} -> Product {product_id}{ctx}")

    # === Orders ===

    def order_created(self, order_id: int, product_count: int, product_type: str, color: str):
        """Log when an order is created."""
        self._log(logging.INFO, f"  ORDER CREATED   | Order {order_id}: {product_count} x {product_type} ({color})")

    def order_complete(self, order_id: int, completion_time: float):
        """Log when an order is completed."""
        self._log(logging.WARNING, f"  ORDER COMPLETE  | Order {order_id} completed at T={completion_time:.1f}")

    def order_status(self, order_id: int, products_packaged: int, total_products: int):
        """Log order progress."""
        self._log(logging.DEBUG, f"  ORDER STATUS    | Order {order_id}: {products_packaged}/{total_products} packaged")

    # === State Snapshots ===

    def state_snapshot(
        self,
        pickup_ready: int,
        pickup_queue: int,
        small_machine_queue: int,
        small_machine_busy: bool,
        big_machine_queue: int,
        big_machine_busy: bool,
        storage_count: int,
        orders_completed: int,
        products_packaged: int
    ):
        """Log a snapshot of the simulation state."""
        self._log(logging.DEBUG, f"  STATE SNAPSHOT:")
        self._log(logging.DEBUG, f"    Pickup: ready={pickup_ready}, queue={pickup_queue}")
        self._log(logging.DEBUG, f"    SmallMachine: queue={small_machine_queue}, busy={small_machine_busy}")
        self._log(logging.DEBUG, f"    BigMachine: queue={big_machine_queue}, busy={big_machine_busy}")
        self._log(logging.DEBUG, f"    Storage: {storage_count} trays")
        self._log(logging.DEBUG, f"    Progress: orders={orders_completed}, products={products_packaged}")

    # === Episode/Step ===

    def step_start(self, step: int):
        """Log start of a simulation step (legacy, use step_begin)."""
        self._log(logging.DEBUG, f"--- Step {step} ---")

    def episode_start(self, episode: int, num_orders: int):
        """Log start of episode."""
        self._log(logging.WARNING, f"{'='*60}")
        self._log(logging.WARNING, f"EPISODE {episode} START | {num_orders} orders")
        self._log(logging.WARNING, f"{'='*60}")

    def episode_end(self, episode: int, total_reward: float, orders_completed: int, total_steps: int = 0):
        """Log end of episode."""
        self._log(logging.WARNING, f"{'='*60}")
        self._log(logging.WARNING, f"EPISODE {episode} END | Reward={total_reward:.2f} | Orders={orders_completed} | Steps={total_steps}")
        self._log(logging.WARNING, f"{'='*60}")

    def episode_summary(
        self,
        episode: int,
        steps: int,
        orders_completed: int,
        total_orders: int,
        products_packaged: int,
        total_products: int,
        reward: float
    ):
        """Log episode summary with order/product progress."""
        self._log(
            logging.WARNING,
            f"Episode {episode:3d} | Steps: {steps:4d} | "
            f"Orders: {orders_completed}/{total_orders} | "
            f"Products: {products_packaged}/{total_products} | "
            f"Reward: {reward:.1f}"
        )


# Global logger instance
logger = SimulationLogger()


def get_logger() -> SimulationLogger:
    """Get the global simulation logger."""
    return logger
