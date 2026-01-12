# MARL-FJSP Grid Visualization

## Overview

The visualization system provides a real-time 4x6 grid world display for monitoring the Multi-Agent Reinforcement Learning Flexible Job Shop Problem (MARL-FJSP) simulation during training.

## Grid Layout

```
     0     1     2     3     4     5
   +-----+-----+-----+-----+-----+-----+
 0 |PICKUP|     |     |     |     |     |
   +-----+-----+-----+-----+-----+-----+
 1 |     | SM  |     |     | BM  |     |
   +-----+-----+-----+-----+-----+-----+
 2 |     |     |     |     |     |     |
   +-----+-----+-----+-----+-----+-----+
 3 |     |     | PKG |     |STOR |     |
   +-----+-----+-----+-----+-----+-----+
```

### Station Positions (from `constants.py`)
| Station | Location Type | Position (row, col) |
|---------|--------------|---------------------|
| Pickup Station | `PICKUP` | (0, 0) |
| Small Machine | `SMALL_MACHINE` | (1, 1) |
| Big Machine | `BIG_MACHINE` | (1, 4) |
| Packaging | `PACKAGING` | (3, 2) |
| Storage | `STORAGE` | (3, 4) |

## Station Colors

Each station type has a distinct background color:

| Station | Color | Hex Code |
|---------|-------|----------|
| Pickup Station | Light Green | `#81C784` |
| Small Machine | Light Blue | `#64B5F6` |
| Big Machine | Light Red | `#E57373` |
| Packaging | Light Purple | `#BA68C8` |
| Storage | Light Gray | `#90A4AE` |

## Agent Action Colors

Each agent's actions are displayed with distinct indicator colors:

| Agent | Action Color | Hex Code |
|-------|-------------|----------|
| Pickup Station | Dark Green | `#2E7D32` |
| AGV | Dark Orange | `#E65100` |
| Small Machine | Dark Blue | `#1565C0` |
| Big Machine | Dark Red | `#C62828` |

## Agent Actions

### Pickup Station
| Action ID | Name | Description |
|-----------|------|-------------|
| 0 | IDLE | No action |
| 1 | LOAD | Load product onto tray |
| 2 | SIGNAL | Signal AGV for pickup |

### AGV
| Action ID | Name | Description |
|-----------|------|-------------|
| 0 | IDLE | No action |
| 1 | GO_PICKUP | Move to pickup station |
| 2 | GO_SM | Move to small machine |
| 3 | GO_BM | Move to big machine |
| 4 | GO_STORAGE | Move to storage |
| 5 | GO_PKG | Move to packaging |
| 6 | PICKUP | Pick up tray |
| 7 | DROP | Drop tray |

### Small Machine / Big Machine
| Action ID | Name | Description |
|-----------|------|-------------|
| 0 | IDLE | No action |
| 1 | PROCESS | Process products |
| 2 | SIGNAL | Signal AGV for pickup |

## Product Visualization

### Order-Based Coloring
Products are colored based on their `order_id` using the golden ratio in HSV color space:

```python
def order_id_to_color(order_id: int) -> str:
    golden_ratio = 0.618033988749895
    hue = (order_id * golden_ratio) % 1.0
    saturation = 0.7
    value = 0.9
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
```

This ensures products from the same order share identical colors, while different orders have visually distinct colors.

### Product States

Products display their processing state through visual styling:

| State | Visual Representation |
|-------|----------------------|
| Unprocessed | Semi-transparent fill with order color, order color edge |
| Processed | White fill, order color edge |
| Packaged | Solid order color fill, black edge |

### Product Locations
Products are drawn as small circles at their current location:
- **Pickup Station**: Products on current tray being loaded
- **Small Machine**: Products in queue + currently processing
- **Big Machine**: Products in queue + currently processing
- **Storage**: Products in stored trays (limited to 5 trays for display)
- **Packaging**: Products in packaging station queues
- **AGV**: Products being transported (shown beside AGV)

## AGV Visualization

- **Default Color**: Orange (`#FF9800`)
- **Carrying Tray**: Darker orange (`#FF5722`)
- **Label**: "AGV" displayed on the patch
- **Position**: Updates based on current grid position

## Information Display

### Action Panel (Right side of grid)
Shows current action for each agent:
- Color indicator box (highlights when action ≠ IDLE)
- Action name text (e.g., "pickup: LOAD")

### Status Bar (Bottom of grid)
```
Step: {step} | Episode: {episode} | Orders: {completed}/{total}
```

### Legend Panel (Right side)
- Station color key
- Product state indicators
- Order color mapping (up to 10 orders displayed)

## Usage

### Enable Visualization
```bash
python train.py --visualize
or
python train.py --timesteps 1000 --heuristic -v --test_only_viz
```

### Programmatic Usage
```python
from a2c import MultiAgentA2C

agent = MultiAgentA2C(env, visualize=True)
agent.learn(total_timesteps=10000)
```

### GridVisualizer API

```python
from visualization import GridVisualizer

# Initialize
viz = GridVisualizer(enabled=True)

# Update each step
viz.update(
    agv_position=(row, col),    # AGV grid position
    actions={'agv': 1, ...},    # Agent actions dict
    step=100,                   # Current step number
    episode=5,                  # Current episode number
    agv_carrying=True,          # Whether AGV has tray
    simulation=sim              # FJSPSimulation object
)

# Keep open after training
viz.keep_open()

# Or close
viz.close()
```

## Data Flow

```
FJSPSimulation
    │
    ├── orders: List[Order]
    │       └── products: List[Product]
    │               ├── order_id → color mapping
    │               ├── is_processed → state display
    │               └── is_packaged → state display
    │
    ├── pickup_station.current_tray.products
    ├── small_machine.tray_queue + current_tray
    ├── big_machine.tray_queue + current_tray
    ├── storage.trays
    ├── packaging_stations[*].tray_queue
    └── agv.carrying_tray.products
            │
            └── GridVisualizer.update()
                    ├── _draw_products_at_location()
                    ├── _draw_agv_products()
                    └── _update_order_legend()
```

## Technical Notes

- **Backend**: Uses `TkAgg` matplotlib backend for interactive display
- **Interactive Mode**: `plt.ion()` enables real-time updates
- **Rendering**: `fig.canvas.draw()` + `flush_events()` + `plt.pause(0.01)`
- **Blocking**: `keep_open()` uses `plt.show(block=True)` to prevent window closure
