from __future__ import annotations

import random
import uuid
from typing import Any

from neurodrive_bench.config import BenchmarkConfig
from neurodrive_bench.contracts import StressProfile, TelemetryFrame
from neurodrive_bench.simulation.factory import build_simulation_backend


class ExpertPIDController:
    """
    Simple PID expert that attempts to stay in the center of the lane and maintain target speed.
    Used for generating synthetic demonstration data.
    """
    def __init__(self, target_speed: float = 12.0) -> None:
        self.target_speed = target_speed
        
        # Steering PID
        self.steer_p = 0.5
        self.steer_d = 0.1
        self.prev_cte = 0.0
        
        # Throttle PID
        self.speed_p = 0.2
        
    def control(self, frame: TelemetryFrame) -> dict[str, float]:
        # Steering (try to reduce lane_offset and heading_error)
        cte = frame.lane_offset + frame.heading_error * 0.5
        steer_diff = cte - self.prev_cte
        self.prev_cte = cte
        
        steering = -(self.steer_p * cte + self.steer_d * steer_diff)
        steering = max(-1.0, min(1.0, steering))
        
        # Throttle/Brake
        speed_error = self.target_speed - frame.speed
        
        # React to obstacles
        if frame.obstacle_distance < 15.0:
            speed_error -= (15.0 - frame.obstacle_distance) * 0.5
            
        throttle = 0.0
        brake = 0.0
        if speed_error > 0:
            throttle = min(1.0, speed_error * self.speed_p)
        else:
            brake = min(1.0, -speed_error * self.speed_p)
            
        return {
            "expert_steering": steering,
            "expert_throttle": throttle,
            "expert_brake": brake
        }


class SyntheticDataCollector:
    """
    Runs episodes using the stub simulation backend and the ExpertPID controller
    to collect demonstration sequences.
    """
    def __init__(self, config: BenchmarkConfig) -> None:
        # Force the backend to be stub for synthetic generation
        self.config = config
        self.config.raw["simulation"]["backend"] = "stub"
        self.backend = build_simulation_backend(self.config)
        self.expert = ExpertPIDController()
        
    def collect(self, num_episodes: int) -> list[dict[str, Any]]:
        self.backend.setup()
        records: list[dict[str, Any]] = []
        
        try:
            print(f"Collecting {num_episodes} synthetic episodes...")
            for ep in range(num_episodes):
                # Vary stress randomly across episodes to get diverse states
                stress_level = random.choice([0.0, 0.25, 0.5, 0.75])
                stress = StressProfile(
                    rain=0.1 * stress_level,
                    fog=0.0,
                    night=False,
                    noise_std=0.02 * stress_level,
                    packet_dropout=0.0,
                    latency_steps=0,
                    sudden_obstacle_probability=0.1 * stress_level,
                    lane_blockage_probability=0.05 * stress_level,
                    traffic_multiplier=1.0 + stress_level
                )
                
                ep_id = str(uuid.uuid4())
                
                # Using the backend directly
                backend = self.backend
                # pylint: disable=protected-access
                backend._current_stress_level = stress_level
                backend._current_stress = stress
                backend._episode_steps = 0
                
                # Randomize initial state slightly
                backend._lane_offset = random.uniform(-0.5, 0.5)
                backend._speed = random.uniform(5.0, 15.0)
                
                for step in range(self.config.max_episode_steps):
                    frame = backend._read_telemetry()
                    
                    # Expert decides action
                    expert_action = self.expert.control(frame)
                    
                    # Record
                    record = {
                        "episode_id": ep_id,
                        "step": step,
                        "stress_level": stress_level,
                        "speed": frame.speed,
                        "lane_offset": frame.lane_offset,
                        "yaw": frame.yaw,
                        "heading_error": frame.heading_error,
                        "obstacle_distance": frame.obstacle_distance,
                        "acceleration": frame.acceleration,
                        "delta_t": frame.delta_t,
                        **expert_action
                    }
                    records.append(record)
                    
                    # Apply action to environment
                    from neurodrive_bench.contracts import ControlCommand, ModelOutput
                    # Provide a dummy model output with the expert's commands to step the environment
                    backend._apply_control(ModelOutput(
                        command=ControlCommand(
                            steering=expert_action["expert_steering"],
                            throttle=expert_action["expert_throttle"],
                            brake=expert_action["expert_brake"]
                        ),
                        uncertainty_score=0.0,
                        adaptation_level=0.0,
                        debug={}
                    ))
                    backend._tick_world()
                    
                if (ep + 1) % 10 == 0:
                    print(f"Collected {ep + 1}/{num_episodes} episodes")
                    
        finally:
            self.backend.teardown()
            
        print(f"Done. Collected {len(records)} total frames.")
        return records
