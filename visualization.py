"""
Minimal Grid World Visualization for MARL-FJSP
==============================================

Displays a 4x6 grid with:
- Fixed station positions (colored squares)
- AGV position (moving)
- Products colored by order ID
- Agent actions with distinct colors
"""

import matplotlib
matplotlib.use('TkAgg')  # Interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Tuple, List, Any
from constants import CONFIG, LOCATION_POSITIONS
from enums.LocationType import LocationType
import colorsys


# Station colors (muted, background)
STATION_COLORS = {
    'pickup_station': '#81C784',   # Light green
    'small_machine': '#64B5F6',    # Light blue
    'big_machine': '#E57373',      # Light red
    'packaging': '#BA68C8',        # Light purple
    'storage': '#90A4AE',          # Light gray
}

# Agent action colors (distinct per agent)
ACTION_COLORS = {
    'pickup_station': '#2E7D32',   # Dark green
    'agv': '#E65100',              # Dark orange
    'small_machine': '#1565C0',    # Dark blue
    'big_machine': '#C62828',      # Dark red
    'packaging_blue_1': '#7B1FA2', # Purple
    'packaging_blue_2': '#7B1FA2', # Purple
    'packaging_red': '#7B1FA2',    # Purple
    'packaging_green': '#7B1FA2',  # Purple
}

# Action names for display
ACTION_NAMES = {
    'pickup_station': {0: 'IDLE', 1: 'LOAD', 2: 'SIGNAL'},
    'agv': {0: 'IDLE', 1: 'GO_PICKUP', 2: 'GO_SM', 3: 'GO_BM',
            4: 'GO_STORAGE', 5: 'GO_PKG', 6: 'PICKUP', 7: 'DROP'},
    'small_machine': {0: 'IDLE', 1: 'PROCESS', 2: 'SIGNAL'},
    'big_machine': {0: 'IDLE', 1: 'PROCESS', 2: 'SIGNAL'},
    'packaging_blue_1': {0: 'IDLE', 1: 'PACKAGE', 2: 'SIGNAL'},
    'packaging_blue_2': {0: 'IDLE', 1: 'PACKAGE', 2: 'SIGNAL'},
    'packaging_red': {0: 'IDLE', 1: 'PACKAGE', 2: 'SIGNAL'},
    'packaging_green': {0: 'IDLE', 1: 'PACKAGE', 2: 'SIGNAL'},
}

# All agent IDs for action display
ALL_AGENTS = ['pickup_station', 'agv', 'small_machine', 'big_machine',
              'packaging_blue_1', 'packaging_blue_2', 'packaging_red', 'packaging_green']


def order_id_to_color(order_id: int) -> str:
    """Generate a distinct color for each order ID using HSV color space."""
    # Use golden ratio to spread hues evenly
    golden_ratio = 0.618033988749895
    hue = (order_id * golden_ratio) % 1.0
    saturation = 0.7
    value = 0.9
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'


class GridVisualizer:
    """Minimal grid visualization for MARL training with product tracking."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if not enabled:
            return

        self.rows = CONFIG['grid_rows']
        self.cols = CONFIG['grid_cols']

        # Create figure with two subplots: grid and legend
        self.fig, (self.ax, self.legend_ax) = plt.subplots(
            1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]}
        )
        self.fig.canvas.manager.set_window_title('MARL-FJSP Grid')

        # Store artists for updating
        self.station_patches = {}
        self.agv_patch = None
        self.action_texts = {}
        self.info_text = None
        self.product_artists = []  # Will be cleared and redrawn each update
        self.order_colors = {}  # Cache order_id -> color mapping

        self._setup_grid()
        self._setup_legend()
        plt.ion()  # Interactive mode
        self.fig.show()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)  # Initial pause to ensure window appears

    def _setup_grid(self):
        """Initialize the grid layout."""
        self.ax.set_xlim(-0.5, self.cols + 2.5)
        self.ax.set_ylim(-0.5, self.rows + 0.8)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()  # Row 0 at top

        # Grid lines
        for i in range(self.rows + 1):
            self.ax.axhline(i - 0.5, color='lightgray', linewidth=0.5)
        for j in range(self.cols + 1):
            self.ax.axvline(j - 0.5, color='lightgray', linewidth=0.5)

        # Draw fixed stations
        station_info = [
            (LocationType.PICKUP, 'PICKUP', 'pickup_station'),
            (LocationType.SMALL_MACHINE, 'SM', 'small_machine'),
            (LocationType.BIG_MACHINE, 'BM', 'big_machine'),
            (LocationType.STORAGE, 'STOR', 'storage'),
            (LocationType.PACKAGING, 'PKG', 'packaging'),
        ]

        for loc_type, label, color_key in station_info:
            row, col = LOCATION_POSITIONS[loc_type]
            color = STATION_COLORS[color_key]

            rect = patches.FancyBboxPatch(
                (col - 0.45, row - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor='black', linewidth=2, alpha=0.6
            )
            self.ax.add_patch(rect)
            self.station_patches[loc_type] = rect

            # Station label
            self.ax.text(col, row - 0.3, label, ha='center', va='center',
                        fontsize=8, fontweight='bold', color='black')

        # AGV patch (will be updated)
        agv_pos = LOCATION_POSITIONS[LocationType.PICKUP]
        self.agv_patch = patches.FancyBboxPatch(
            (agv_pos[1] - 0.2, agv_pos[0] - 0.15), 0.4, 0.3,
            boxstyle="round,pad=0.02",
            facecolor='#FF9800', edgecolor='black', linewidth=2, zorder=10
        )
        self.ax.add_patch(self.agv_patch)
        self.agv_label = self.ax.text(agv_pos[1], agv_pos[0], 'AGV', ha='center', va='center',
                                       fontsize=7, fontweight='bold', color='white', zorder=11)

        # Action display area (right side of grid)
        self.action_texts['title'] = self.ax.text(
            self.cols + 0.3, -0.3, 'ACTIONS', fontsize=10, fontweight='bold'
        )

        y_offset = 0.1
        for agent_id in ALL_AGENTS:
            color = ACTION_COLORS.get(agent_id, '#666666')
            # Action indicator box
            indicator = patches.Rectangle(
                (self.cols + 0.2, y_offset - 0.08), 0.12, 0.18,
                facecolor=color, edgecolor='black', linewidth=1
            )
            self.ax.add_patch(indicator)
            self.action_texts[f'{agent_id}_indicator'] = indicator

            # Shorter display name for packaging agents
            display_name = agent_id[:8] if 'packaging' in agent_id else agent_id[:6]

            # Action text
            self.action_texts[agent_id] = self.ax.text(
                self.cols + 0.4, y_offset, f'{display_name}: --',
                fontsize=6, color='black', fontweight='bold', va='center'
            )
            y_offset += 0.32

        # Info text (bottom)
        self.info_text = self.ax.text(
            0, self.rows + 0.5, 'Step: 0 | Episode: 0 | Orders: 0',
            fontsize=10, color='black', fontweight='bold'
        )

        self.ax.set_title('MARL-FJSP Grid World', fontsize=12, fontweight='bold')
        self.ax.axis('off')

    def _setup_legend(self):
        """Setup the legend panel."""
        self.legend_ax.set_xlim(0, 1)
        self.legend_ax.set_ylim(0, 1)
        self.legend_ax.axis('off')
        self.legend_ax.set_title('Product Legend', fontsize=10, fontweight='bold')

        # Station legend
        y = 0.95
        self.legend_ax.text(0.05, y, 'Stations:', fontsize=9, fontweight='bold')
        y -= 0.06
        for name, color in STATION_COLORS.items():
            rect = patches.Rectangle((0.05, y - 0.02), 0.08, 0.04,
                                     facecolor=color, edgecolor='black')
            self.legend_ax.add_patch(rect)
            self.legend_ax.text(0.15, y, name.replace('_', ' ').title(),
                               fontsize=7, va='center')
            y -= 0.05

        # Tray legend
        y -= 0.03
        self.legend_ax.text(0.05, y, 'Trays:', fontsize=9, fontweight='bold')
        y -= 0.06
        tray_rect = patches.FancyBboxPatch(
            (0.05, y - 0.02), 0.12, 0.04,
            boxstyle="round,pad=0.01",
            facecolor='#FFFDE7', edgecolor='#666', linewidth=1
        )
        self.legend_ax.add_patch(tray_rect)
        self.legend_ax.text(0.19, y, 'Tray (holds products)', fontsize=7, va='center')
        y -= 0.06

        # Product state legend
        y -= 0.02
        self.legend_ax.text(0.05, y, 'Product States:', fontsize=9, fontweight='bold')
        y -= 0.06

        states = [
            ('Unprocessed', '#E91E63', '#E91E63', 0.5),
            ('Processed', 'white', '#E91E63', 0.9),
            ('Packaged', '#E91E63', 'black', 1.0),
        ]
        for label, fill, edge, alpha in states:
            circle = patches.Circle((0.09, y), 0.02, facecolor=fill, edgecolor=edge,
                                    linewidth=1, alpha=alpha)
            self.legend_ax.add_patch(circle)
            self.legend_ax.text(0.15, y, label, fontsize=7, va='center')
            y -= 0.05

        # Order colors section (will be updated dynamically)
        y -= 0.03
        self.legend_ax.text(0.05, y, 'Orders (by color):', fontsize=9, fontweight='bold')
        self.order_legend_y_start = y - 0.06
        self.order_legend_artists = []

    def _get_order_color(self, order_id: int) -> str:
        """Get or create a color for an order ID."""
        if order_id not in self.order_colors:
            self.order_colors[order_id] = order_id_to_color(order_id)
        return self.order_colors[order_id]

    def _draw_product(self, px: float, py: float, product: Any, size: float = 0.06, zorder: int = 6):
        """Draw a single product circle."""
        order_color = self._get_order_color(product.order_id)

        # Visual state based on processing status
        if product.is_packaged:
            edge_color = 'black'
            fill_color = order_color
            alpha = 1.0
        elif product.is_processed:
            edge_color = order_color
            fill_color = 'white'
            alpha = 0.9
        else:
            edge_color = order_color
            fill_color = order_color
            alpha = 0.5

        circle = patches.Circle(
            (px, py), size,
            facecolor=fill_color, edgecolor=edge_color,
            linewidth=1.5, alpha=alpha, zorder=zorder
        )
        self.ax.add_patch(circle)
        self.product_artists.append(circle)

    def _draw_tray(self, tx: float, ty: float, tray: Any, zorder: int = 5):
        """Draw a tray as a rectangle with products inside."""
        # Tray rectangle
        tray_width = 0.35
        tray_height = 0.25

        # Tray border is always gray
        tray_rect = patches.FancyBboxPatch(
            (tx - tray_width/2, ty - tray_height/2), tray_width, tray_height,
            boxstyle="round,pad=0.02",
            facecolor='#FFFDE7', edgecolor='#666666',
            linewidth=2, alpha=0.9, zorder=zorder
        )
        self.ax.add_patch(tray_rect)
        self.product_artists.append(tray_rect)

        # Tray ID label
        tray_label = self.ax.text(
            tx, ty - tray_height/2 - 0.05, f'T{tray.id % 100}',
            ha='center', va='top', fontsize=5, color='gray', zorder=zorder+1
        )
        self.product_artists.append(tray_label)

        # Draw products inside tray (max 5)
        for i, product in enumerate(tray.products[:5]):
            px = tx - 0.12 + (i % 3) * 0.12
            py = ty + 0.02 - (i // 3) * 0.12
            self._draw_product(px, py, product, size=0.04, zorder=zorder+2)

    def _draw_trays_at_location(self, loc_type: LocationType, trays: List[Any],
                                 offset_x: float = 0, offset_y: float = 0.2):
        """Draw trays at a station location."""
        if not trays:
            return

        row, col = LOCATION_POSITIONS[loc_type]
        # Arrange trays in a row below station
        for i, tray in enumerate(trays[:4]):  # Max 4 trays shown
            tx = col - 0.4 + (i % 2) * 0.45 + offset_x
            ty = row + 0.15 + (i // 2) * 0.35 + offset_y
            self._draw_tray(tx, ty, tray)

    def _draw_agv_tray(self, tray: Any, agv_pos: Tuple[int, int]):
        """Draw tray being carried by AGV."""
        if not tray:
            return

        row, col = agv_pos
        # Draw tray next to AGV
        self._draw_tray(col + 0.4, row, tray, zorder=11)

    def _update_order_legend(self, simulation=None):
        """Update the order color legend with completion status."""
        # Clear old legend entries
        for artist in self.order_legend_artists:
            artist.remove()
        self.order_legend_artists = []

        # Get completed order IDs
        completed_order_ids = set()
        if simulation is not None:
            completed_order_ids = {order.id for order in simulation.completed_orders}

        y = self.order_legend_y_start
        for order_id, color in sorted(self.order_colors.items())[:10]:  # Show max 10
            is_complete = order_id in completed_order_ids

            rect = patches.Rectangle((0.05, y - 0.015), 0.06, 0.03,
                                     facecolor=color, edgecolor='black', linewidth=0.5)
            self.legend_ax.add_patch(rect)
            self.order_legend_artists.append(rect)

            # Add checkmark for completed orders
            if is_complete:
                label = f'Order {order_id} DONE'
                text = self.legend_ax.text(0.13, y, label, fontsize=7, va='center',
                                          fontweight='bold', color='green')
            else:
                text = self.legend_ax.text(0.13, y, f'Order {order_id}', fontsize=7, va='center')
            self.order_legend_artists.append(text)
            y -= 0.04

    def update(self,
               agv_position: Tuple[int, int],
               actions: Dict[str, int],
               step: int,
               episode: int,
               agv_carrying: bool = False,
               simulation: Any = None):
        """
        Update visualization for current step.

        Parameters
        ----------
        agv_position : Tuple[int, int]
            Current (row, col) position of AGV
        actions : Dict[str, int]
            Action taken by each agent
        step : int
            Current step number
        episode : int
            Current episode number
        agv_carrying : bool
            Whether AGV is carrying a tray
        simulation : FJSPSimulation
            The simulation object to get product data from
        """
        if not self.enabled:
            return

        # Clear previous product artists
        for artist in self.product_artists:
            artist.remove()
        self.product_artists = []

        # Update AGV position
        row, col = agv_position
        self.agv_patch.set_x(col - 0.2)
        self.agv_patch.set_y(row - 0.15)
        self.agv_label.set_position((col, row))

        # Change AGV color if carrying
        if agv_carrying:
            self.agv_patch.set_facecolor('#FF5722')  # Darker orange
        else:
            self.agv_patch.set_facecolor('#FF9800')

        # Update action texts with color indicators for all agents
        for agent_id in ALL_AGENTS:
            if agent_id in actions:
                action_num = actions[agent_id]
                action_name = ACTION_NAMES.get(agent_id, {}).get(action_num, str(action_num))
                display_name = agent_id[:8] if 'packaging' in agent_id else agent_id[:6]

                # Highlight indicator if action is not IDLE
                indicator = self.action_texts.get(f'{agent_id}_indicator')
                if indicator:
                    if action_num != 0:  # Not IDLE
                        indicator.set_alpha(1.0)
                        indicator.set_linewidth(2)
                    else:
                        indicator.set_alpha(0.3)
                        indicator.set_linewidth(1)

                self.action_texts[agent_id].set_text(f'{display_name}: {action_name}')

        # Draw trays and products at each location
        if simulation is not None:
            # Trays at pickup station
            pickup_trays = []
            if simulation.pickup_station.current_tray:
                pickup_trays.append(simulation.pickup_station.current_tray)
            pickup_trays.extend(simulation.pickup_station.ready_trays)
            self._draw_trays_at_location(LocationType.PICKUP, pickup_trays)

            # Trays at small machine
            sm_trays = list(simulation.small_machine.tray_queue)
            if simulation.small_machine.current_tray:
                sm_trays.append(simulation.small_machine.current_tray)
            sm_trays.extend(simulation.small_machine.ready_trays)
            self._draw_trays_at_location(LocationType.SMALL_MACHINE, sm_trays)

            # Trays at big machine
            bm_trays = list(simulation.big_machine.tray_queue)
            if simulation.big_machine.current_tray:
                bm_trays.append(simulation.big_machine.current_tray)
            bm_trays.extend(simulation.big_machine.ready_trays)
            self._draw_trays_at_location(LocationType.BIG_MACHINE, bm_trays)

            # Trays in storage
            self._draw_trays_at_location(LocationType.STORAGE, simulation.storage.trays[:4])

            # Products at packaging (no trays, individual products)
            pkg_products = []
            for station in simulation.packaging_stations.values():
                pkg_products.extend(station.product_queue[:4])
            # Draw packaging products directly
            row, col = LOCATION_POSITIONS[LocationType.PACKAGING]
            for i, product in enumerate(pkg_products[:8]):
                px = col - 0.3 + (i % 4) * 0.15
                py = row + 0.2 + (i // 4) * 0.15
                self._draw_product(px, py, product, size=0.05)

            # Tray on AGV
            if simulation.agv.carrying_tray:
                self._draw_agv_tray(simulation.agv.carrying_tray, agv_position)

            # Update order legend with completion status
            self._update_order_legend(simulation)

            # Update info with order count
            completed = len(simulation.completed_orders)
            total = len(simulation.orders)
            self.info_text.set_text(
                f'Step: {step} | Episode: {episode} | Orders: {completed}/{total}'
            )
        else:
            self.info_text.set_text(f'Step: {step} | Episode: {episode}')

        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # Small pause for visibility

    def keep_open(self):
        """Keep the visualization window open after training ends."""
        if self.enabled:
            plt.ioff()
            plt.show(block=True)

    def close(self):
        """Close the visualization window."""
        if self.enabled:
            plt.ioff()
            plt.close(self.fig)
