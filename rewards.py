"""
Reward shaping functions for Pokémon Yellow RL training.
"""

# Reward scale constants (keep step-to-step rewards small for PPO stability)
# Move to Environment config later if needed
STEP_PENALTY = -0.001
NEW_TILE_REWARD = 0.02
BADGE_DELTA_REWARD = 1.0
STUCK_PENALTY = -0.02
STUCK_THRESHOLD_STEPS = 10
LEVEL_UP_REWARD = 0.2  # keep small
STATUS_PENALTY = -0.01  # very small per-step

import os
from dotenv import load_dotenv
from memory_map import decode_player_position, decode_in_battle, decode_badge_count, decode_party_stats, get_opponent_roster_count, decode_escaped_from_battle, decode_party_levels, decode_party_statuses

class RewardShaper:
    """
    Handles reward computation and shaping for Pokémon Yellow training.
    """
    
    def __init__(self):
        self.previous_position = None
        self.steps_at_position = 0
        self.tiles_visited = set()
        self.previous_badge_count = 0
        self.previous_battle_wins = 0
        self.previous_battle_losses = 0
        self.total_tiles_visited = 0
        self._prev_party_levels = [0]*6
        # Battle state tracking
        self._prev_in_battle = False
        self._battle_start_party_hp_sum = None
        self._prev_opponent_roster_count = 0
        
    def compute_reward(self, ram_data: bytes, action: int, info: dict) -> float:
        """
        Compute reward based on current RAM state and action.
        
        Args:
            ram_data: Current RAM state
            action: Action taken
            info: Episode information
            
        Returns:
            float: Computed reward
        """
        reward = 0.0
        
        # Base penalty for inaction
        reward += STEP_PENALTY
        
        # Exploration reward
        exploration_reward = self._compute_exploration_reward(ram_data)
        reward += exploration_reward
        
        # Battle rewards
        battle_reward = self._compute_battle_reward(ram_data, info)
        reward += battle_reward
        
        # Progress rewards
        progress_reward = self._compute_progress_reward(ram_data)
        reward += progress_reward
        
        # Anti-stuck reward
        anti_stuck_reward = self._compute_anti_stuck_reward(ram_data)
        reward += anti_stuck_reward
        
        reward += self._compute_level_reward(ram_data)
        reward += self._compute_status_penalty(ram_data)

        return reward
    
    def _compute_exploration_reward(self, ram_data: bytes) -> float:
        """Reward for visiting new tiles."""
        position = decode_player_position(ram_data)
        map_id, x, y = position
        
        # Create tile identifier
        tile_id = (map_id, x, y)
        
        # Reward for new tiles
        if tile_id not in self.tiles_visited:
            self.tiles_visited.add(tile_id)
            self.total_tiles_visited += 1
            return NEW_TILE_REWARD
            
        return 0.0
    
    def _compute_battle_reward(self, ram_data: bytes, info: dict = None) -> float:
        """Return +1.0 on a detected battle win, -1.0 on a detected battle loss, else 0.0.

        Detection strategy (simple + robust):
        - Track `in_battle` transitions.
        - When a battle ends (prev True -> current False):
            - If all party HP is 0.0 => loss
            - Else if opponent roster count transitioned from >0 to 0 => win
            - Else => 0.0 (likely ran away / transition)

        Notes:
        - This avoids rewarding fleeing as a win.
        - You can refine later with explicit victory flags or money/exp deltas.
        """
        try:
            # Prefer explicit battle-end metadata from the env (Option B)
            if isinstance(info, dict) and info.get('battle_ended', False):
                # If we believe this was a run-away (either via escaped flag or draw result), never reward it.
                if bool(info.get('ran_away', False)) or bool(info.get('escaped_from_battle', False)):
                    return 0.0

                br = int(info.get('battle_result', -1))
                known = bool(info.get('battle_result_known', False))

                # Only score outcomes when the env tells us the result is known.
                if known:
                    if br == 1:
                        self.previous_battle_losses += 1
                        return -1.0
                    if br == 0:
                        self.previous_battle_wins += 1
                        return 1.0
                    # br == 2 is draw/run-away
                    return 0.0

                # Unknown result -> fall through to heuristic (no early return)

            # --- Fallback: previous heuristic when battle_result isn't available ---
            curr_in_battle = bool(decode_in_battle(ram_data))
            party_hp = decode_party_stats(ram_data)
            party_sum = float(sum(party_hp))
            curr_roster = get_opponent_roster_count(ram_data)

            outcome = 0.0

            # Battle started
            if (not self._prev_in_battle) and curr_in_battle:
                self._battle_start_party_hp_sum = party_sum

            # Battle ended
            if self._prev_in_battle and (not curr_in_battle):
                # If an escape item/move was used, never treat as a win.
                if decode_escaped_from_battle(ram_data):
                    outcome = 0.0
                elif party_sum <= 0.0:
                    outcome = -1.0
                    self.previous_battle_losses += 1
                else:
                    prev_roster = int(getattr(self, "_prev_opponent_roster_count", 0))
                    # Win only if roster was present and is now cleared
                    if prev_roster > 0 and int(curr_roster) == 0:
                        outcome = 1.0
                        self.previous_battle_wins += 1
                    else:
                        outcome = 0.0

            # Update trackers
            self._prev_in_battle = curr_in_battle
            self._prev_opponent_roster_count = curr_roster

            return outcome
        except Exception:
            # Keep the training loop resilient
            return 0.0
    
    def _compute_progress_reward(self, ram_data: bytes) -> float:
        """Reward for game progress like badges."""
        current_badge_count = decode_badge_count(ram_data)
        reward = 0.0
        
        # Reward for earning badges
        if current_badge_count > self.previous_badge_count:
            delta = int(current_badge_count - self.previous_badge_count)
            reward += float(delta) * BADGE_DELTA_REWARD
            self.previous_badge_count = int(current_badge_count)
            
        return reward
    
    def _compute_anti_stuck_reward(self, ram_data: bytes) -> float:
        """Penalize staying in same position too long."""
        position = decode_player_position(ram_data)
        map_id, x, y = position
        
        # Check if position is the same as previous
        if self.previous_position == (map_id, x, y):
            self.steps_at_position += 1
            # Penalize being stuck
            if self.steps_at_position > STUCK_THRESHOLD_STEPS:
                return STUCK_PENALTY
        else:
            # Reset counter when position changes
            self.steps_at_position = 0
            self.previous_position = (map_id, x, y)
            
        return 0.0
    
    def reset(self, seed=None):
        """Reset reward tracking state."""
        self.previous_position = None
        self.steps_at_position = 0
        self.tiles_visited = set()
        self.previous_badge_count = 0
        self.previous_battle_wins = 0
        self.previous_battle_losses = 0
        self.total_tiles_visited = 0
        self._prev_in_battle = False
        self._battle_start_party_hp_sum = None
        self._prev_opponent_roster_count = 0
        self._prev_party_levels = [0]*6
        # Return empty info dict to match Gymnasium reset signature
        return {}

    def _compute_level_reward(self, ram_data: bytes) -> float:
        lv = decode_party_levels(ram_data)
        r = 0.0
        for i in range(min(len(lv), 6)):
            if lv[i] > self._prev_party_levels[i]:
                r += LEVEL_UP_REWARD * float(lv[i] - self._prev_party_levels[i])
        self._prev_party_levels = lv
        return r

    def _compute_status_penalty(self, ram_data: bytes) -> float:
        st = decode_party_statuses(ram_data)
        # If any party mon has a non-zero status, apply a small penalty
        return STATUS_PENALTY if any((int(s) & 0xFF) != 0 for s in st) else 0.0    
    
# Phase-based reward shaping
class PhaseRewardShaper(RewardShaper):
    """
    Reward shaper that adapts based on training phase.
    """
    
    def __init__(self, phase: int = 1):
        super().__init__()
        self.phase = phase
        # Hybrid (Option C): manual phase, automatic tuning within a phase
        # Multipliers default to 1.0 and can be adjusted live by callbacks.
        self.dynamic = {
            'exploration': 1.0,
            'battle_win': 1.0,
            'battle_loss': 1.0,
            'badge': 1.0,
            'stuck': 1.0,
        }
        # Phase-specific reward weights
        self.weights = {
            1: {  # Movement sandbox
                'exploration': 1.0,
                'battle_win': 0.0,
                'battle_loss': 0.0,
                'badge': 0.0,
                'stuck': 1.0,
            },
            2: {  # Early route
                'exploration': 1.0,
                'battle_win': 0.10,
                'battle_loss': -0.20,
                'badge': 0.0,
                'stuck': 1.0,
            },
            3: {  # First battles
                'exploration': 0.7,
                'battle_win': 0.25,
                'battle_loss': -0.35,
                'badge': 0.0,
                'stuck': 1.0,
            },
            4: {  # Badge pursuit
                'exploration': 0.4,
                'battle_win': 0.20,
                'battle_loss': -0.30,
                'badge': 1.0,
                'stuck': 1.0,
            }
        }
        
    def compute_reward(self, ram_data: bytes, action: int, info: dict) -> float:
        reward = 0.0
        reward += STEP_PENALTY

        w = self.weights.get(int(self.phase), self.weights[1])  # <-- guard

        exploration_reward = self._compute_exploration_reward(ram_data)
        reward += exploration_reward * w['exploration'] * self.dynamic.get('exploration', 1.0)

        battle_outcome = self._compute_battle_reward(ram_data, info)
        if battle_outcome > 0:
            reward += w['battle_win'] * self.dynamic.get('battle_win', 1.0)
        elif battle_outcome < 0:
            reward += w['battle_loss'] * self.dynamic.get('battle_loss', 1.0)

        progress_reward = self._compute_progress_reward(ram_data)
        reward += progress_reward * w['badge'] * self.dynamic.get('badge', 1.0)

        anti_stuck_reward = self._compute_anti_stuck_reward(ram_data)
        reward += anti_stuck_reward * w['stuck'] * self.dynamic.get('stuck', 1.0)

        reward += self._compute_level_reward(ram_data)
        reward += self._compute_status_penalty(ram_data)

        return reward

    def set_phase(self, phase: int) -> None:
        """Manually set the current phase (Option C uses manual gating)."""
        try:
            phase = int(phase)
        except Exception:
            return
        if phase in self.weights:
            self.phase = phase

    def get_phase(self) -> int:
        try:
            return int(self.phase)
        except Exception:
            return 1

    def set_dynamic_multiplier(self, key: str, value: float) -> None:
        """Set a dynamic multiplier for a reward component (exploration, stuck, etc.)."""
        if not isinstance(key, str):
            return
        if key not in self.dynamic:
            return
        try:
            v = float(value)
        except Exception:
            return
        # Defensive clamp
        if v < 0.0:
            v = 0.0
        if v > 5.0:
            v = 5.0
        self.dynamic[key] = v

    def set_dynamic_multipliers(self, updates: dict) -> None:
        """Batch update dynamic multipliers."""
        if not isinstance(updates, dict):
            return
        for k, v in updates.items():
            self.set_dynamic_multiplier(str(k), v)

# Curriculum-based reward shaping
class CurriculumRewardShaper(RewardShaper):
    """
    Reward shaper that implements curriculum learning.
    """
    
    def __init__(self):
        super().__init__()
        self.current_stage = 1
        self.stage_thresholds = {
            1: 100,  # Movement sandbox
            2: 500,  # Early route
            3: 1000, # First battles
            4: 2000, # Badge pursuit
        }
        
    def update_stage(self, tiles_visited: int):
        """Update training stage based on progress."""
        for stage, threshold in self.stage_thresholds.items():
            if tiles_visited >= threshold:
                self.current_stage = stage
            else:
                break
                
    def compute_reward(self, ram_data: bytes, action: int, info: dict) -> float:
        """Compute reward with curriculum-based weights."""
        # This would be implemented based on current stage
        return super().compute_reward(ram_data, action, info)