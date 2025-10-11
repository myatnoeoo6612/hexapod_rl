# # Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause
# """Package containing asset and sensor configurations."""

# import os
# import toml

# # Conveniences to other module directories via relative paths
# ISAACLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
# """Path to the extension source directory."""

# ISAACLAB_ASSETS_DATA_DIR = os.path.join(ISAACLAB_ASSETS_EXT_DIR, "data")
# """Path to the extension data directory."""

# ISAACLAB_ASSETS_METADATA = toml.load(os.path.join(ISAACLAB_ASSETS_EXT_DIR, "config", "extension.toml"))
# """Extension metadata dictionary parsed from the extension.toml file."""

# # Configure the module-level variables
# __version__ = ISAACLAB_ASSETS_METADATA["package"]["version"]

# from .robots import *
# from .sensors import *
# Copyright (c) 2022-2025, My Hexapod Project Developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Robots module for my_hexapod_walk_rl.
Contains robot asset configs.
"""

from .my_hexapod_walk_rl import MY_HEXAPOD_WALK_RL_CFG

__all__ = ["MY_HEXAPOD_WALK_RL_CFG"]
