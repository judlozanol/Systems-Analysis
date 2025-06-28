import gfootball.env as football_env
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import time

# ------------------------------------- 
# Ball-focused custom reward function
# ------------------------------------- 
class BallFocusedRewards:
    def __init__(self):
        # Ball-centered rewards
        self.GOAL_REWARD = 1000.0           # Max reward per goal
        self.GOAL_CONCEDED_PENALTY = -1000.0  # Goal conceded penalty
        self.BALL_STOLEN_PENALTY = -20.0      # Ball stolen penalty
        self.BACKWARD_LONG_PASS_PENALTY = -3.0 # Backward long pass penalty
        
        self.BALL_POSSESSION_REWARD = 0.3  # Reward for having the ball
        self.BALL_LOST_PENALTY = -2.0      # Penalty for losing the ball
        
        # Directional rewards
        self.FORWARD_MOVEMENT_REWARD = 0.4   # Reward for moving toward opponent's goal
        self.BACKWARD_MOVEMENT_PENALTY = -0.2 # Penalty for moving backward
        self.ATTACK_ZONE_BONUS = 0.5         # Bonus for being in attack zone
        
        # Ball action rewards
        self.SUCCESSFUL_PASS_REWARD = 1.5   # Successful pass
        self.FORWARD_PASS_BONUS = 1.0       # Bonus for forward pass
        self.SHOT_REWARD = 2.0              # Reward for shooting
        self.GOOD_SHOT_BONUS = 1.2          # Bonus for good shot
        self.BAD_SHOT_PENALTY = -0.3        # Small penalty for bad shot
        
        # Time penalty system
        self.BALL_HOLDING_TOO_LONG_PENALTY = -0.5  # Penalty for holding ball too long
        self.MAX_BALL_HOLDING_TIME = 15     # Max allowed time with ball (steps)
        self.PROGRESSIVE_PENALTY_START = 10 # When to start progressive penalty
        
        # Ball recovery rewards
        self.BALL_RECOVERY_REWARD = 1.2     # Recovering possession
        self.INTERCEPTION_REWARD = 1.8      # Intercepting opponent's pass
        
        # Proximity to ball rewards (when we don't have it)
        self.CLOSE_TO_BALL_REWARD = 0.3     # Being close to the ball
        self.MOVING_TO_BALL_REWARD = 0.2    # Moving toward the ball
        
        # Field zones (based on documentation: field spans from -1 to 1 in X)
        self.OWN_GOAL_AREA = (-1.0, -0.8)
        self.OWN_DEFENSIVE_AREA = (-1.0, -0.3)
        self.OWN_HALF = (-1.0, 0.0)
        self.OPPONENT_HALF = (0.0, 1.0)
        self.OPPONENT_AREA = (0.3, 1.0)
        
    def parse_simple115_observation(self, obs):
        """Parse simple115v2 observation format (115 floats)"""
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
        
        # Ball ownership (3 floats: one-hot encoding)
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
        
    def get_field_zone(self, x_pos):
        """Determine which zone of the field the position is in"""
        if x_pos <= -0.8:
            return "own_goal_area"
        elif x_pos <= -0.3:
            return "own_defensive_area"
        elif x_pos <= 0.0:
            return "own_half"
        elif x_pos <= 0.3:
            return "opponent_half"
        else:
            return "opponent_area"
    
    def get_ball_ownership_team(self, ball_ownership):
        """Get which team owns the ball from one-hot encoding"""
        # ball_ownership = [no_owner, left_team, right_team]
        if ball_ownership[1] == 1:
            return 0  # Left team (us)
        elif ball_ownership[2] == 1:
            return 1  # Right team (opponent)
        else:
            return -1  # No owner
    
    def get_active_player_idx(self, active_player):
        """Get the index of the active player"""
        return np.argmax(active_player)
    
    def _estimate_player_role(self, player_idx, player_x, player_y):
        """Estimate player role based on index and position"""
        if player_idx == 0:
            return "goalkeeper"
        elif player_idx in [1, 2]:
            return "center_back"
        elif player_idx in [3, 4]:
            return "full_back"
        elif player_idx in [5, 6]:
            return "central_midfielder"
        elif player_idx in [7, 8]:
            return "winger"
        else:  # player_idx in [9, 10]
            return "striker"
    
    def calculate_ball_focused_reward(self, obs, action, info=None, prev_obs=None, ball_holding_time=0, prev_score=None, current_score=None):
        """
        Ball-focused reward system with time penalty and corrected directions
        """
        parsed = self.parse_simple115_observation(obs)
        reward = 0.0
        
        # Detect conceded goals
        if prev_score is not None and current_score is not None:
            if current_score[1] > prev_score[1]:  # Opponent scored
                print(f"Goal conceded! Massive penalty: {self.GOAL_CONCEDED_PENALTY}")
                return self.GOAL_CONCEDED_PENALTY
        
        # Detect goals scored
        if info and 'score_reward' in info and info['score_reward'] > 0:
            print(f"Goal scored! Reward: {self.GOAL_REWARD}")
            return self.GOAL_REWARD
        
        # Get player and ball information
        active_idx = self.get_active_player_idx(parsed['active_player'])
        player_pos = parsed['left_team_pos'][active_idx]
        player_x, player_y = player_pos[0], player_pos[1]
        ball_x, ball_y = parsed['ball_pos'][0], parsed['ball_pos'][1]
        ball_owned_team = self.get_ball_ownership_team(parsed['ball_ownership'])
        
        # Get player role and field zone
        player_role = self._estimate_player_role(active_idx, player_x, player_y)
        field_zone = self.get_field_zone(player_x)
        
        # Ball possession rewards and actions
        if ball_owned_team == 0:  # We have the ball
            reward += self.BALL_POSSESSION_REWARD
            
            # Directional rewards
            reward += self._reward_directional_movement(action, player_x, player_y, ball_owned_team)
            
            # Penalty for holding ball too long
            if ball_holding_time >= self.PROGRESSIVE_PENALTY_START:
                # Progressive penalty that increases with time
                time_penalty = self.BALL_HOLDING_TOO_LONG_PENALTY * (ball_holding_time - self.PROGRESSIVE_PENALTY_START + 1)
                reward += time_penalty
                
                if ball_holding_time % 5 == 0:  # Debug every 5 steps
                    print(f"Holding ball for {ball_holding_time} steps, penalty: {time_penalty:.2f}")
            
            # Big rewards for ball actions (to encourage passing/shooting)
            action_reward = self._reward_ball_actions(action, player_role, field_zone, 
                                                    player_x, player_y, ball_x, ball_y, ball_holding_time)
            reward += action_reward
            
            # Extra penalty if holding ball too long without action
            if ball_holding_time > self.MAX_BALL_HOLDING_TIME:
                extreme_penalty = self.BALL_HOLDING_TOO_LONG_PENALTY * 3
                reward += extreme_penalty
                print(f"Ball held too long! ({ball_holding_time} steps) Penalty: {extreme_penalty:.2f}")
        
        else:  # We don't have the ball
            # Directional rewards without ball
            reward += self._reward_directional_movement(action, player_x, player_y, ball_owned_team)
            
            # Rewards for trying to recover the ball
            reward += self._reward_ball_recovery(parsed, action, active_idx, prev_obs)
        
        # Detect if we lost the ball (increased penalties)
        if prev_obs is not None:
            prev_parsed = self.parse_simple115_observation(prev_obs)
            prev_ball_owned = self.get_ball_ownership_team(prev_parsed['ball_ownership'])
            if prev_ball_owned == 0 and ball_owned_team != 0:  # Lost the ball
                reward += self.BALL_LOST_PENALTY + self.BALL_STOLEN_PENALTY
                print(f"Ball stolen! Total penalty: {self.BALL_LOST_PENALTY + self.BALL_STOLEN_PENALTY}")
        
        return reward
    
    def _reward_directional_movement(self, action, player_x, player_y, has_ball):
        """Directional rewards to correct movement"""
        reward = 0.0
        
        # Only apply directional rewards for movements (actions 0-7)
        if action not in range(8):
            return reward
        
        # Directional action mapping:
        # 0: Idle, 1: Left, 2: Top-Left, 3: Top, 4: Top-Right, 
        # 5: Right, 6: Bottom-Right, 7: Bottom
        
        # Rewards for moving toward opponent's goal (positive X)
        if action in [4, 5, 6]:  # Top-Right, Right, Bottom-Right (forward)
            if has_ball:
                reward += self.FORWARD_MOVEMENT_REWARD 
                print(f"Advancing with ball! +{self.FORWARD_MOVEMENT_REWARD:.2f}")
            else:
                reward += self.FORWARD_MOVEMENT_REWARD *0.25
        
        # Penalties for moving backward (negative X)
        elif action in [1, 2]:  # Left, Top-Left (backward)
            if player_x > -0.5:  # Only penalize if not very far back defending
                penalty = self.BACKWARD_MOVEMENT_PENALTY
                if has_ball:
                    penalty *= 3  # Triple penalty with ball
                    print(f"⬇️ Moving backward with ball! {penalty:.2f}")
                reward += penalty
        
        # Bonus for being in attack zone
        if player_x > 0.3:  # In opponent's attack zone
            reward += self.ATTACK_ZONE_BONUS
            if has_ball:
                reward += self.ATTACK_ZONE_BONUS * 0.5  # Extra bonus with ball
        
        return reward
    
    def _reward_ball_actions(self, action, player_role, field_zone, player_x, player_y, ball_x, ball_y, ball_holding_time):
        """Rewards for actions when we have the ball"""
        reward = 0.0
        
        # Big rewards for passes and shots to encourage action
        if action in [9, 10, 11]:  # Passes (long, high, short)
            base_pass_reward = self.SUCCESSFUL_PASS_REWARD
            
            # Massive penalty for backward long passes
            if action in [9, 10]:  # Long passes
                if player_x > 0.2:  # If in offensive field
                    # Heavily penalize backward long passes from offensive zone
                    reward += base_pass_reward + self.BACKWARD_LONG_PASS_PENALTY
                    print(f"Long backward pass from offensive zone! Penalty: {self.BACKWARD_LONG_PASS_PENALTY:.2f}")
                elif player_x > -0.3:  # Midfield forward
                    reward += base_pass_reward + self.FORWARD_PASS_BONUS
                    print(f"Long forward pass! +{base_pass_reward + self.FORWARD_PASS_BONUS:.2f}")
                else:  # Very defensive field - allow long passes
                    reward += base_pass_reward + (self.FORWARD_PASS_BONUS * 0.7)
            else:  
                # Short passes are always safe
                reward += base_pass_reward
                print(f"Safe short pass! +{base_pass_reward:.2f}")
            
            # Extra bonus if holding ball for long time
            if ball_holding_time >= self.PROGRESSIVE_PENALTY_START:
                urgency_bonus = 0.5 * (ball_holding_time - self.PROGRESSIVE_PENALTY_START + 1)
                reward += urgency_bonus
                print(f"Urgent pass after {ball_holding_time} steps! Bonus: +{urgency_bonus:.2f}")
        
        elif action == 12:  # Shot
            base_shot_reward = self.SHOT_REWARD
            
            # Evaluate shot quality ONLY by distance (not considering player position)
            distance_to_goal = abs(1.0 - player_x)
            
            if distance_to_goal < 0.3:  # Very close to goal
                reward += base_shot_reward + self.GOOD_SHOT_BONUS
                print(f"Close shot! +{base_shot_reward + self.GOOD_SHOT_BONUS:.2f}")
            elif distance_to_goal < 0.6:  # Medium distance
                reward += base_shot_reward + (self.GOOD_SHOT_BONUS * 0.5)
            else:  # Far
                reward += base_shot_reward + self.BAD_SHOT_PENALTY
            
            # Extra bonus for shooting if holding ball for long time
            if ball_holding_time >= self.PROGRESSIVE_PENALTY_START:
                urgency_bonus = 0.3 * (ball_holding_time - self.PROGRESSIVE_PENALTY_START + 1)
                reward += urgency_bonus
                print(f"Urgent shot after {ball_holding_time} steps! Bonus: +{urgency_bonus:.2f}")
        
        elif action == 13:  # Sprint with ball
            # Reward for sprinting toward opponent's goal
            if player_x > 0:  # In opponent's field
                reward += self.SUCCESSFUL_PASS_REWARD * 0.8
            else:  # In our field
                reward += self.SUCCESSFUL_PASS_REWARD * 0.4
        
        return reward
    
    def _reward_ball_recovery(self, parsed, action, active_idx, prev_obs):
        """Rewards for trying to recover the ball"""
        reward = 0.0
        
        player_pos = parsed['left_team_pos'][active_idx]
        ball_pos = parsed['ball_pos'][:2]
        
        # Distance to ball
        distance_to_ball = np.linalg.norm(player_pos - ball_pos)
        
        # Reward for being close to ball
        if distance_to_ball < 0.05:  # Very close
            reward += self.CLOSE_TO_BALL_REWARD * 2
        elif distance_to_ball < 0.1:  # Close
            reward += self.CLOSE_TO_BALL_REWARD
        elif distance_to_ball < 0.2:  # Relatively close
            reward += self.CLOSE_TO_BALL_REWARD * 0.5
        
        # Reward for moving toward ball
        if prev_obs is not None and action in range(8):  # Directional movements
            prev_parsed = self.parse_simple115_observation(prev_obs)
            prev_player_pos = prev_parsed['left_team_pos'][active_idx]
            prev_distance = np.linalg.norm(prev_player_pos - ball_pos)
            
            if distance_to_ball < prev_distance:  # We're getting closer to ball
                reward += self.MOVING_TO_BALL_REWARD
        
        # Bonus for defensive actions near ball
        if distance_to_ball < 0.1 and action == 13:  # Sprint toward ball
            reward += self.CLOSE_TO_BALL_REWARD
        
        return reward

# ------------------------------------- 
# Custom wrapper with ball-focused rewards - FIXED
# ------------------------------------- 
class BallFocusedWrapper(gym.Wrapper):
    def __init__(self, env):
        super(BallFocusedWrapper, self).__init__(env)
        self.previous_obs = None
        self.reward_calculator = BallFocusedRewards()
        self.step_count = 0
        self.ball_holding_time = 0  # Time with ball counter
        self.previous_ball_owner = -1  # To detect possession changes
        self.previous_score = [0, 0]  # [our_goals, opponent_goals]
        
    def reset(self):
        obs = self.env.reset()
        self.previous_obs = obs.copy() if obs is not None else None
        self.step_count = 0
        self.ball_holding_time = 0
        self.previous_ball_owner = -1
        self.previous_score = [0, 0]
        return obs
    
    def step(self, action):
        obs, original_reward, done, info = self.env.step(action)
        self.step_count += 1
        
        # Get current score
        current_score = [0, 0]
        if info and 'score' in info:
            current_score = info['score']
        
        # Update ball holding time counter
        if obs is not None:
            parsed = self.reward_calculator.parse_simple115_observation(obs)
            current_ball_owner = self.reward_calculator.get_ball_ownership_team(parsed['ball_ownership'])
            
            if current_ball_owner == 0:  # We have the ball
                if self.previous_ball_owner == 0:
                    self.ball_holding_time += 1  # Increment time
                else:
                    self.ball_holding_time = 1  # Just got the ball
            else:
                self.ball_holding_time = 0  # Reset counter
            
            self.previous_ball_owner = current_ball_owner
        
        # Calculate custom reward
        custom_reward = 0.0
        if obs is not None:
            try:
                custom_reward = self.reward_calculator.calculate_ball_focused_reward(
                    obs, action, info, self.previous_obs, self.ball_holding_time,
                    self.previous_score, current_score
                )
            except Exception as e:
                print(f"Warning: Error calculating custom reward: {e}")
                custom_reward = 0.0
        
        # Combine rewards
        if abs(custom_reward) > 500:  # Goals or big penalties
            total_reward = custom_reward  # Use only custom reward for important events
        else:
            # For normal actions, give more weight to custom rewards
            total_reward = original_reward + (custom_reward * 0.7)
        
        # Debug every 50 steps
        if self.step_count % 50 == 0:
            ball_owned = current_ball_owner if obs is not None else -1
            print(f"Step {self.step_count}: Ball owned by team {ball_owned}, "
                  f"Holding time: {self.ball_holding_time}, Custom reward: {custom_reward:.3f}, "
                  f"Score: {current_score}")
        
        self.previous_obs = obs.copy() if obs is not None else None
        self.previous_score = current_score.copy()
        return obs, total_reward, done, info

# ------------------------------------- 
# Create environment with ball-focused wrapper
# ------------------------------------- 
def create_ball_focused_env():
    env = football_env.create_environment(
        env_name='11_vs_11_hard_stochastic',
        representation='simple115v2',
        number_of_left_players_agent_controls=1,
        render=False,
        write_goal_dumps=False,
        write_full_episode_dumps=False
    )
    return BallFocusedWrapper(env)

# ------------------------------------- 
# Train PPO agent with ball-focused rewards
# ------------------------------------- 
def train_ball_focused_agent():
    print("Starting ball-focused training...")
    env = DummyVecEnv([create_ball_focused_env])
    
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
        tensorboard_log="./ppo_gfootball_ball_focused_tensorboard/"
    )
    
    print("Training with ball-focused rewards...")
    print("Priority: Ball possession > Ball actions > Direction > Positioning")
    model.learn(total_timesteps=100_000)
    model.save("ppo_gfootball_ball_focused")
    print("Ball-focused training completed and model saved!")

# ------------------------------------- 
# Evaluate ball-focused agent
# ------------------------------------- 
def evaluate_ball_focused_agent(num_episodes=5, render=True):
    print("Starting ball-focused evaluation...")
    
    env = football_env.create_environment(
        env_name='11_vs_11_hard_stochastic',
        representation='simple115v2',
        number_of_left_players_agent_controls=1,
        render=render,
        write_goal_dumps=False,
        write_full_episode_dumps=False
    )
    env = BallFocusedWrapper(env)
    
    try:
        model = PPO.load("ppo_gfootball_ball_focused")
        print("Ball-focused model loaded successfully!")
    except FileNotFoundError:
        print("No ball-focused model found. Please train first.")
        return
    
    episode_rewards = []
    episode_goals = []
    possession_stats = []
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}")
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        goals_scored = 0
        ball_possession_steps = 0
        total_steps = 0
        
        while not done and step_count < 3000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Track possession
            parsed = env.reward_calculator.parse_simple115_observation(obs)
            ball_owned = env.reward_calculator.get_ball_ownership_team(parsed['ball_ownership'])
            if ball_owned == 0:
                ball_possession_steps += 1
            total_steps += 1
            
            total_reward += reward
            step_count += 1
            
            # Check for goals
            if 'score_reward' in info and info['score_reward'] > 0:
                goals_scored += 1
                print(f"⚽ GOAL! Step {step_count}")
            
            if step_count % 500 == 0:
                possession_pct = (ball_possession_steps / total_steps) * 100
                print(f"Step {step_count}, Reward: {total_reward:.3f}, Possession: {possession_pct:.1f}%")
            
            if render:
                time.sleep(0.01)
        
        possession_percentage = (ball_possession_steps / total_steps) * 100 if total_steps > 0 else 0
        
        episode_rewards.append(total_reward)
        episode_goals.append(goals_scored)
        possession_stats.append(possession_percentage)
        
        print(f"📊 Episode {episode + 1} Results:")
        print(f"Total Reward: {total_reward:.3f}")
        print(f"Goals Scored: {goals_scored}")
        print(f"Ball Possession: {possession_percentage:.1f}%")
        print(f"Steps: {step_count}")
    
    # Summary statistics
    print(f"\nBall-Focused Evaluation Summary ({num_episodes} episodes):")
    print(f"Average Reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"Average Goals: {np.mean(episode_goals):.2f} ± {np.std(episode_goals):.2f}")
    print(f"Average Possession: {np.mean(possession_stats):.1f}% ± {np.std(possession_stats):.1f}%")
    print(f"Best Episode: {np.max(episode_rewards):.3f} reward, {np.max(episode_goals)} goals")
    
    env.close()

# ------------------------------------- 
# Main execution
# ------------------------------------- 
if __name__ == "__main__":
    print("⚽ Google Research Football - Ball-Focused Reward Training")
    print("=" * 65)
    
    mode = input("Choose mode (train/evaluate/quick_test): ").strip().lower()
    
    if mode == "train":
        train_ball_focused_agent()
    elif mode == "evaluate":
        evaluate_ball_focused_agent()
    elif mode == "quick_test":
        print("Running quick test (1 episode, no rendering)...")
        evaluate_ball_focused_agent(num_episodes=1, render=False)
    else:
        print("Invalid mode. Please choose 'train', 'evaluate', or 'quick_test'")