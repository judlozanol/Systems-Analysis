import gfootball.env as football_env
import gym
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# =====================================
# Reward System for Contextual Actions
# =====================================
class ContextualActionRewardSystem:
    def __init__(self):
        # Defining the field zones
        self.ZONES = {
            'own_goal': (-1.0, -0.9),      # Own net
            'own_penalty': (-0.9, -0.7),   # Own penalty area
            'own_defense': (-0.7, -0.4),   # Own defense area
            'own_midfield': (-0.4, -0.1),  # Own half field
            'center': (-0.1, 0.1),         # Half field
            'opp_midfield': (0.1, 0.4),    # Opponent half field
            'opp_defense': (0.4, 0.7),     # Opponent defense area
            'opp_penalty': (0.7, 0.9),    # Opponent penalty area
            'opp_goal': (0.9, 1.0)         # Opponent net
        }
        
        # Basic Rewards
        self.GOAL_REWARD = 1000.0
        self.GOAL_CONCEDED_PENALTY = -1000.0
        self.BALL_RECOVERY_REWARD = 20.0
        self.BALL_LOST_PENALTY = -25.0
        
        # Ball Retention Penalty System
        self.ball_holding_count = 0
        self.current_ball_holder = None
        
        # Ball Retention Limits
        self.HOLDING_LIMITS = {
            'own_goal': 8,          
            'own_penalty': 10,      
            'own_defense': 15,      
            'own_midfield': 20,     
            'center': 25,           
            'opp_midfield': 30,     
            'opp_defense': 35,      
            'opp_penalty': 40,      
            'opp_goal': 45          
        }
        
        # Contextual Action Rewards
        self.ACTION_REWARDS = {
            # Passes by Direction and Context
            'pass_forward': {
                'base_reward': 5.0,
                'distance_bonus': 2.0,      # Distance bonus
                'pressure_bonus': 3.0,      # Pressure bonus
                'danger_zone_bonus': 4.0,   # Bonus when progressing from danger zone
                'offensive_bonus': 2.0      # Bonus in offensive zone
            },
            
            'pass_backward': {
                'base_reward': 1.0,
                'safety_bonus': 2.0,        # Safety bonus
                'buildup_bonus': 1.5,       # Build-up play bonus
                'danger_penalty': -3.0,     # Penalty in offensive position
                'pressure_penalty': -2.0    # Pressure penalty
            },
            
            'pass_lateral': {
                'base_reward': 2.0,
                'circulation_bonus': 1.0,   # Ball circulation bonus
                'pressure_bonus': 1.5,     # Pressure bonus
                'time_penalty': -1.0       # Penalty for lack of progression
            },
            
            # === SHOTS BY CONTEXT ===
            'shot': {
                'close_range': 15.0,        # Close range shot
                'medium_range': 8.0,        # Medium range shot
                'long_range': 3.0,          # Long range shot
                'angle_bonus': 5.0,         # Good angle bonus
                'pressure_bonus': 2.0,      # Pressure bonus
                'bad_position_penalty': -8.0  # Bad position penalty
            },
            
            # === DRIBBLING BY CONTEXT ===
            'dribble': {
                'progression_bonus': 4.0,   # Progression bonus
                'danger_zone_bonus': 3.0,   # Danger zone bonus
                'pressure_bonus': 2.0,      # Pressure bonus
                'safe_zone_penalty': -1.0,  # Safe zone penalty
                'holding_penalty': -2.0     # Excessive holding penalty
            },
            
            # === DEFENSIVE ACTIONS ===
            'tackle': {
                'successful_bonus': 8.0,    # Successful tackle bonus
                'danger_zone_bonus': 5.0,   # Danger zone bonus
                'failed_penalty': -3.0      # Failed tackle penalty
            },
            
            # === OFF-THE-BALL MOVEMENT ===
            'movement': {
                'approach_ball_bonus': 2.0,     # Ball approach bonus
                'space_creation_bonus': 1.5,    # Space creation bonus
                'support_bonus': 1.0,           # Support play bonus
                'static_penalty': -0.5          # Static play penalty
            }
        }
        
        # GAME PHASE MULTIPLIERS
        self.PHASE_MULTIPLIERS = {
            'attack': {
                'pass_forward': 1.5,
                'pass_backward': 0.5,
                'shot': 1.8,
                'dribble': 1.3
            },
            'defense': {
                'pass_forward': 1.2,
                'pass_backward': 1.0,
                'tackle': 1.5,
                'movement': 1.3
            },
            'transition': {
                'pass_forward': 1.8,
                'dribble': 1.5,
                'movement': 1.4
            }
        }
    
    def get_zone(self, x_position):
        """Determine field zone based on X position"""
        for zone, (min_x, max_x) in self.ZONES.items():
            if min_x <= x_position <= max_x:
                return zone
        return 'center'
    
    def get_player_role(self, player_idx):
        """Determine player role based on index"""
        if player_idx == 0:
            return 'goalkeeper'
        elif player_idx <= 4:
            return 'defense'
        elif player_idx <= 7:
            return 'midfield'
        else:
            return 'attack'
    
    def parse_simple115_observation(self, obs):
        """Parse simple115v2 observation"""
        idx = 0
        
        # Left team positions (22 floats: 11 players * 2 coords)
        left_team_pos = obs[idx:idx+22].reshape(11, 2)
        idx += 22
        
        # Left team directions (22 floats: 11 players * 2 coords)
        left_team_dir = obs[idx:idx+22].reshape(11, 2)
        idx += 22
        
        # Right team positions (22 floats: 11 players * 2 coords)
        right_team_pos = obs[idx:idx+22].reshape(11, 2)
        idx += 22
        
        # Right team directions (22 floats: 11 players * 2 coords)
        right_team_dir = obs[idx:idx+22].reshape(11, 2)
        idx += 22
        
        # Ball position (3 floats: x, y, z)
        ball_pos = obs[idx:idx+3]
        idx += 3
        
        # Ball direction (3 floats: x, y, z)
        ball_dir = obs[idx:idx+3]
        idx += 3
        
        # Ball possession (3 floats: one-hot encoding)
        ball_ownership = obs[idx:idx+3]
        idx += 3
        
        # Active player (11 floats: one-hot encoding)
        active_player = obs[idx:idx+11]
        idx += 11
        
        # Game mode (7 floats: one-hot encoding)
        game_mode = obs[idx:idx+7]
        
        return {
            'left_team_pos': left_team_pos,
            'left_team_dir': left_team_dir,
            'right_team_pos': right_team_pos,
            'right_team_dir': right_team_dir,
            'ball_pos': ball_pos,
            'ball_dir': ball_dir,
            'ball_ownership': ball_ownership,
            'active_player': active_player,
            'game_mode': game_mode
        }
    
    def get_ball_ownership_team(self, ball_ownership):
        """Get which team has ball possession"""
        if ball_ownership[1] == 1:
            return 0  # Left team (our team)
        elif ball_ownership[2] == 1:
            return 1  # Right team (opponent)
        else:
            return -1  # No possession
    
    def get_active_player_idx(self, active_player):
        """Get active player index"""
        return np.argmax(active_player)
    
    def detect_game_phase(self, parsed_obs, ball_owned_team):
        """Detect game phase"""
        ball_pos = parsed_obs['ball_pos']
        ball_x = ball_pos[0]
        
        if ball_owned_team == 0:  # We have possession
            if ball_x > 0.3:
                return 'attack'
            elif ball_x < -0.3:
                return 'buildup'
            else:
                return 'transition'
        elif ball_owned_team == 1:  # Opponent has possession
            return 'defense'
        else:
            return 'transition'
    
    def calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def is_under_pressure(self, player_pos, opponent_positions):
        """Determine if player is under pressure"""
        min_distance = float('inf')
        for opp_pos in opponent_positions:
            distance = self.calculate_distance(player_pos, opp_pos)
            min_distance = min(min_distance, distance)
        return min_distance < 0.1  # Pressure threshold
    
    def get_pass_direction(self, player_pos, ball_pos, action):
        """Determine pass direction"""
        if action in [9, 10]:  # Long forward passes
            return 'forward'
        elif action == 11:  # Short pass - analyze direction
            # Simplified: if in own half, likely forward
            if player_pos[0] < 0:
                return 'forward'
            else:
                return 'lateral'
        return 'lateral'
    
    def get_shot_quality(self, player_pos, goal_pos=(1.0, 0.0)):
        """Evaluate shot quality"""
        distance = self.calculate_distance(player_pos, goal_pos)
        angle = abs(player_pos[1])  # Angle relative to center
        
        if distance < 0.2:
            return 'close_range'
        elif distance < 0.5:
            return 'medium_range'
        else:
            return 'long_range'
    
    def update_ball_holding(self, active_player_idx, has_ball):
        """Update ball holding counter"""
        if has_ball:
            if self.current_ball_holder == active_player_idx:
                self.ball_holding_count += 1
            else:
                self.current_ball_holder = active_player_idx
                self.ball_holding_count = 1
        else:
            self.ball_holding_count = 0
            self.current_ball_holder = None
    
    def get_ball_holding_penalty(self, player_pos):
        """Calculate excessive holding penalty"""
        if self.ball_holding_count == 0:
            return 0.0
        
        current_zone = self.get_zone(player_pos[0])
        limit = self.HOLDING_LIMITS.get(current_zone, 20)
        
        if self.ball_holding_count > limit:
            # Progressive penalty
            excess = self.ball_holding_count - limit
            return -0.5 * excess * (1 + excess * 0.1)
        
        return 0.0
    
    def calculate_contextual_reward(self, obs, action, info=None, prev_obs=None):
        """Calculate contextual action reward"""
        parsed = self.parse_simple115_observation(obs)
        
        # Basic information
        active_idx = self.get_active_player_idx(parsed['active_player'])
        player_pos = parsed['left_team_pos'][active_idx]
        ball_pos = parsed['ball_pos']
        ball_owned_team = self.get_ball_ownership_team(parsed['ball_ownership'])
        opponent_positions = parsed['right_team_pos']
        
        # Game states
        has_ball = (ball_owned_team == 0)
        game_phase = self.detect_game_phase(parsed, ball_owned_team)
        under_pressure = self.is_under_pressure(player_pos, opponent_positions)
        current_zone = self.get_zone(player_pos[0])
        
        # Update holding counter
        self.update_ball_holding(active_idx, has_ball)
        
        reward = 0.0
        
        # FUNDAMENTAL REWARDS
        if info and 'score_reward' in info:
            if info['score_reward'] > 0:
                return self.GOAL_REWARD
            elif info['score_reward'] < 0:
                return self.GOAL_CONCEDED_PENALTY
        
        # POSSESSION CHANGES
        if prev_obs is not None:
            prev_parsed = self.parse_simple115_observation(prev_obs)
            prev_ball_owned = self.get_ball_ownership_team(prev_parsed['ball_ownership'])
            
            if prev_ball_owned != 0 and ball_owned_team == 0:
                reward += self.BALL_RECOVERY_REWARD
            elif prev_ball_owned == 0 and ball_owned_team != 0:
                reward += self.BALL_LOST_PENALTY
        
        # EXCESSIVE HOLDING PENALTY
        if has_ball:
            holding_penalty = self.get_ball_holding_penalty(player_pos)
            reward += holding_penalty
        
        # ACTION REWARDS
        if has_ball:
            # PASSES
            if action in [9, 10, 11]:
                pass_direction = self.get_pass_direction(player_pos, ball_pos, action)
                action_key = f'pass_{pass_direction}'
                
                if action_key in self.ACTION_REWARDS:
                    action_reward = self.ACTION_REWARDS[action_key]['base_reward']
                    
                    # Context bonus
                    if under_pressure:
                        if pass_direction == 'forward':
                            action_reward += self.ACTION_REWARDS[action_key].get('pressure_bonus', 0)
                        elif pass_direction == 'lateral':
                            action_reward += self.ACTION_REWARDS[action_key].get('pressure_bonus', 0)
                    
                    # Zone bonus/penalty
                    if pass_direction == 'forward':
                        if current_zone in ['own_goal', 'own_penalty', 'own_defense']:
                            action_reward += self.ACTION_REWARDS[action_key].get('danger_zone_bonus', 0)
                        elif current_zone in ['opp_defense', 'opp_penalty']:
                            action_reward += self.ACTION_REWARDS[action_key].get('offensive_bonus', 0)
                    
                    elif pass_direction == 'backward':
                        if current_zone in ['opp_defense', 'opp_penalty', 'opp_goal']:
                            action_reward += self.ACTION_REWARDS[action_key].get('danger_penalty', 0)
                        if under_pressure:
                            action_reward += self.ACTION_REWARDS[action_key].get('pressure_penalty', 0)
                    
                    # Apply phase multiplier
                    if game_phase in self.PHASE_MULTIPLIERS:
                        multiplier = self.PHASE_MULTIPLIERS[game_phase].get(action_key, 1.0)
                        action_reward *= multiplier
                    
                    reward += action_reward
            
            # SHOTS
            elif action == 12:
                shot_quality = self.get_shot_quality(player_pos)
                shot_reward = self.ACTION_REWARDS['shot'].get(shot_quality, 0)
                
                # Angle bonus (simplified)
                if abs(player_pos[1]) < 0.3:  # Near center
                    shot_reward += self.ACTION_REWARDS['shot'].get('angle_bonus', 0)
                
                # Pressure bonus
                if under_pressure:
                    shot_reward += self.ACTION_REWARDS['shot'].get('pressure_bonus', 0)
                
                # Bad position penalty
                if current_zone in ['own_goal', 'own_penalty', 'own_defense']:
                    shot_reward += self.ACTION_REWARDS['shot'].get('bad_position_penalty', 0)
                
                # Apply phase multiplier
                if game_phase in self.PHASE_MULTIPLIERS:
                    multiplier = self.PHASE_MULTIPLIERS[game_phase].get('shot', 1.0)
                    shot_reward *= multiplier
                
                reward += shot_reward
            
            # DRIBBLING
            elif action == 13:
                dribble_reward = self.ACTION_REWARDS['dribble']['base_reward']
                
                # Progression bonus
                if current_zone in ['opp_midfield', 'opp_defense', 'opp_penalty']:
                    dribble_reward += self.ACTION_REWARDS['dribble'].get('progression_bonus', 0)
                
                # Pressure bonus
                if under_pressure:
                    dribble_reward += self.ACTION_REWARDS['dribble'].get('pressure_bonus', 0)
                
                # Safe zone penalty
                if current_zone in ['own_defense', 'own_midfield'] and not under_pressure:
                    dribble_reward += self.ACTION_REWARDS['dribble'].get('safe_zone_penalty', 0)
                
                # Holding penalty
                if self.ball_holding_count > 15:
                    dribble_reward += self.ACTION_REWARDS['dribble'].get('holding_penalty', 0)
                
                # Apply phase multiplier
                if game_phase in self.PHASE_MULTIPLIERS:
                    multiplier = self.PHASE_MULTIPLIERS[game_phase].get('dribble', 1.0)
                    dribble_reward *= multiplier
                
                reward += dribble_reward
        
        else:
            # OFF-THE-BALL ACTIONS
            # Movement toward ball
            ball_distance = self.calculate_distance(player_pos, ball_pos[:2])
            if ball_distance < 0.3:  # Approaching ball
                approach_reward = self.ACTION_REWARDS['movement'].get('approach_ball_bonus', 0)
                reward += approach_reward
        
        return reward


# =====================================
# CONTEXTUAL WRAPPER
# =====================================
class ContextualWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ContextualWrapper, self).__init__(env)
        self.previous_obs = None
        self.reward_system = ContextualActionRewardSystem()
        self.step_count = 0
        
    def reset(self):
        obs = self.env.reset()
        self.previous_obs = obs.copy() if obs is not None else None
        self.step_count = 0
        self.reward_system.ball_holding_count = 0
        self.reward_system.current_ball_holder = None
        return obs
    
    def step(self, action):
        obs, original_reward, done, info = self.env.step(action)
        self.step_count += 1
        
        # Calculate contextual reward
        contextual_reward = 0.0
        if obs is not None:
            try:
                contextual_reward = self.reward_system.calculate_contextual_reward(
                    obs, action, info, self.previous_obs
                )
            except Exception as e:
                print(f"Warning: Error calculating contextual reward: {e}")
                contextual_reward = 0.0
        
        # Combine rewards
        if abs(contextual_reward) > 500:  # Important events
            total_reward = contextual_reward
        else:
            total_reward = original_reward + contextual_reward
        
        # Debug every 100 steps
        if self.step_count % 100 == 0:
            if obs is not None:
                parsed = self.reward_system.parse_simple115_observation(obs)
                active_idx = self.reward_system.get_active_player_idx(parsed['active_player'])
                player_pos = parsed['left_team_pos'][active_idx]
                ball_owned = self.reward_system.get_ball_ownership_team(parsed['ball_ownership'])
                holding_count = self.reward_system.ball_holding_count
                
                print(f"Step {self.step_count}: Player {active_idx}, "
                      f"Ball owned: {ball_owned}, Holding: {holding_count}, "
                      f"Action: {action}, Reward: {contextual_reward:.3f}")
        
        self.previous_obs = obs.copy() if obs is not None else None
        return obs, total_reward, done, info


# =====================================
# ENVIRONMENT CREATION FUNCTION
# =====================================
def create_contextual_env(renderBool):
    """Create environment with contextual wrapper"""
    env = football_env.create_environment(
        env_name='11_vs_11_stochastic',  # / academy_pass_and_shoot_with_keeper
        representation='simple115v2',
        number_of_left_players_agent_controls=1,
        render=renderBool,
        write_goal_dumps=False,
        write_full_episode_dumps=False
    )
    return ContextualWrapper(env)

# ------------------------------------- 
# Train PPO agent with contextual rewards
# ------------------------------------- 
def train_contextual_agent():
    print("Starting contextual action training...")
    env = create_contextual_env(False)
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./ppo_gfootball_contextual_tensorboard/"
    )
    
    print("Training with contextual action rewards...")
    print("Focus: Smart Actions > Ball Movement > No Ball Holding")
    model.learn(total_timesteps=100_000)
    model.save("ppo_gfootball_copy")
    print("Contextual training completed and model saved!")
    env.close()

# ------------------------------------- 
# Evaluate contextual agent
# ------------------------------------- 
def evaluate_contextual_agent(num_episodes=5, render=True):
    print("Starting contextual evaluation...")
    
    env = create_contextual_env(render)
    
    try:
        model = PPO.load("ppo_gfootball_copy")
        print("Contextual model loaded successfully!")
    except FileNotFoundError:
        print("No contextual model found. Please train first.")
        return
    
    episode_rewards = []
    episode_goals = []
    ball_holding_stats = []
    action_stats = []
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}")
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        goals_scored = 0
        max_holding = 0
        avg_holding = 0
        holding_violations = 0
        forward_passes = 0
        shots_taken = 0
        
        while not done and step_count < 3000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Tracking statistics
            current_holding = env.reward_system.ball_holding_count
            max_holding = max(max_holding, current_holding)
            avg_holding += current_holding
            
            if current_holding > 25:  # General limit
                holding_violations += 1
            
            # Count actions
            if action in [9, 10]:  # Long passes
                forward_passes += 1
            elif action == 12:  # Shots
                shots_taken += 1
            
            total_reward += reward
            step_count += 1
            
            # Check for goals
            if 'score_reward' in info and info['score_reward'] > 0:
                goals_scored += 1
                print(f" GOAL! Step {step_count}")
            
            if step_count % 500 == 0:
                avg_hold = avg_holding / step_count
                print(f"Step {step_count}, Reward: {total_reward:.3f}, "
                      f"Max Holding: {max_holding}, Avg Holding: {avg_hold:.1f}, "
                      f"Violations: {holding_violations}")
            
            if render:
                time.sleep(0.01)
        
        avg_holding = avg_holding / step_count if step_count > 0 else 0
        
        episode_rewards.append(total_reward)
        episode_goals.append(goals_scored)
        ball_holding_stats.append({
            'max_holding': max_holding,
            'avg_holding': avg_holding,
            'violations': holding_violations
        })
        action_stats.append({
            'forward_passes': forward_passes,
            'shots_taken': shots_taken
        })
        
        print(f"Episode {episode + 1} Results:")
        print(f"Total Reward: {total_reward:.3f}")
        print(f"Goals Scored: {goals_scored}")
        print(f"Max Ball Holding: {max_holding} steps")
        print(f"Avg Ball Holding: {avg_holding:.1f} steps")
        print(f"Holding Violations: {holding_violations}")
        print(f"Forward Passes: {forward_passes}")
        print(f"Shots Taken: {shots_taken}")
    
    # Summary statistics
    print(f"\nContextual Evaluation Summary ({num_episodes} episodes):")
    print(f"Average Reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"Average Goals: {np.mean(episode_goals):.2f} ± {np.std(episode_goals):.2f}")
    
    avg_max_holding = np.mean([s['max_holding'] for s in ball_holding_stats])
    avg_avg_holding = np.mean([s['avg_holding'] for s in ball_holding_stats])
    avg_violations = np.mean([s['violations'] for s in ball_holding_stats])
    
    print(f"Average Max Holding: {avg_max_holding:.1f} steps")
    print(f"Average Avg Holding: {avg_avg_holding:.1f} steps")
    print(f"Average Violations: {avg_violations:.1f}")
    
    avg_forward_passes = np.mean([s['forward_passes'] for s in action_stats])
    avg_shots = np.mean([s['shots_taken'] for s in action_stats])
    
    print(f"Average Forward Passes: {avg_forward_passes:.1f}")
    print(f"Average Shots: {avg_shots:.1f}")
    
    env.close()

# ------------------------------------- 
# Main execution
# ------------------------------------- 
if __name__ == "__main__":
    print("Google Research Football - Contextual Action Rewards")
    print("=" * 65)
    
    mode = input("Choose mode (train/evaluate/quick_test): ").strip().lower()
    
    if mode == "train":
        train_contextual_agent()
    elif mode == "evaluate":
        evaluate_contextual_agent()
    elif mode == "quick_test":
        print("Running quick test (1 episode, no rendering)...")
        evaluate_contextual_agent(num_episodes=1, render=False)
    else:
        print("Invalid mode. Please choose 'train', 'evaluate', or 'quick_test'")