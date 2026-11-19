"""Hierarchical crossing environment. Import ONLY after AppLauncher.

Owns physics stepping/reset so terminal rewards see pre-reset state. Native
locomotion reward/termination managers are not used to train the behavior actor.
"""
from dataclasses import replace
from types import SimpleNamespace
import torch
from tensordict import TensorDict
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensorCfg
from isaaclab_physx.sensors.contact_sensor.contact_sensor import ContactSensor
from isaaclab.sim.utils.queries import get_all_matching_child_prims
from pxr import UsdPhysics
from h1_finetune.runtime import validate_contract
from h1_finetune.control import clamp_commands
from h1_finetune.numerics import finite
from crossing_geometry import CrossingConfig
from crossing_observations import set_scripted_vehicle_velocity
from crossing_rewards import CrossingRewardTracker
from play_crossing_finetuned import build_config
from behavior_control import BatchedTraffic, speed_from_action, heading_command, behavior_observation
from vehicle_motion import TrafficConfig


class VehicleRobotContactSensor(ContactSensor):
    """One vehicle body per env, one filter entry per robot rigid body."""

    def _initialize_impl(self):
        root = "/World/envs/env_0/Robot"
        bodies = get_all_matching_child_prims(
            root, predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI))
        if not bodies:
            raise RuntimeError("No H1 rigid bodies found for collision filtering")
        # PhysX requires each filter to resolve to exactly one body per env.
        # A single Robot/.* filter would resolve to num_envs * num_bodies.
        self.cfg.filter_prim_paths_expr = [
            "/World/envs/env_.*/Robot" + str(body.GetPath())[len(root):]
            for body in bodies]
        super()._initialize_impl()


class CrossingBehaviorEnv:
    num_actions = 1

    def __init__(self, num_envs, device, seed, settings, rewards, bundle, manifest, headless=True,
                 traffic_config=TrafficConfig(), perception_config=None, perception_debug=False,
                 radar_config=None, radar_output=None, radar_observations=False,
                 oracle_observations=False, groundtruth23=False, privileged_critic=False):
        self.settings, self.reward_cfg = settings, rewards
        self.device, self.num_envs = device, num_envs
        self.headless = headless
        self.appearance = None
        self.perception = None
        self.radar = None
        self.trajectory_recorder = None
        self.radar_observations = radar_observations
        self.oracle_observations = oracle_observations
        self.groundtruth23 = groundtruth23
        self.privileged_critic = privileged_critic
        if privileged_critic and (not radar_observations or perception_config is None
                                  or groundtruth23 or oracle_observations):
            raise ValueError("Privileged critic requires genuine camera/radar actor observations")
        if groundtruth23 and (perception_config is not None or radar_config is not None
                              or radar_observations or oracle_observations):
            raise ValueError("Groundtruth23 is a sensor-free input source, separate from oracle evaluation")
        if oracle_observations and (not radar_observations or perception_config is None):
            raise ValueError("Oracle evaluation requires the camera_radar 23D schema")
        if radar_observations and radar_config is None:
            raise ValueError("Radar observations require the radar sensor")
        if radar_config is not None and (perception_config is None or radar_output is None
                                        or radar_config.sample_hz != perception_config.frequency_hz):
            raise ValueError("Radar requires camera observations, matching capture frequency and an output folder")
        if radar_config is not None:
            from radar_batch import validate_sensor_spacing
            validate_sensor_spacing(num_envs, settings.env_spacing_m, perception_config, radar_config)
        self.perception_config = perception_config
        self.sim_time = 0.
        self.physics_steps = 0
        self.observation_dim = 23 if radar_observations or groundtruth23 else 22 if perception_config else 19
        self.critic_observation_dim = 29 if privileged_critic else self.observation_dim
        self.layout = replace(CrossingConfig(), road_length=110., goal_position=(0., 4.6, .017))
        self.cfg = build_config(SimpleNamespace(device=device, raised_sidewalks=False, h1_camera=False), "E", self.layout)
        self.cfg.seed = seed
        self.cfg.scene.num_envs = num_envs
        self.cfg.scene.env_spacing = settings.env_spacing_m
        self.cfg.episode_length_s = settings.episode_seconds
        # One sensing body per env, filtering ONLY robot partners (not ground).
        self.cfg.scene.vehicle.spawn.activate_contact_sensors = True
        self.cfg.scene.vehicle_contact = ContactSensorCfg(
            class_type=VehicleRobotContactSensor,
            prim_path="{ENV_REGEX_NS}/Vehicle", update_period=0.0, history_length=1,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*"],
            max_contact_data_count_per_prim=128)
        if perception_config is not None:
            from h1_rgbd import add_cameras
            add_cameras(self.cfg.scene, perception_config)
            # Remove the ground-truth vehicle observation term entirely in camera mode.
            self.cfg.observations.crossing = None
        self.base = ManagerBasedRLEnv(cfg=self.cfg)
        try:
            self.contract = validate_contract(self.base, self.cfg)
            if self.contract != manifest["contract"]:
                raise ValueError("Frozen locomotion contract does not match crossing environment")
            if self.base._physics_handles_decimation:
                raise ValueError("This loop requires per-substep stepping (PhysX), not backend decimation")
            self.locomotion = torch.jit.load(str(bundle / "policy.pt"), map_location=device).eval()
            for parameter in self.locomotion.parameters():
                parameter.requires_grad_(False)
            self.robot = self.base.scene["robot"]
            self.vehicle = self.base.scene["vehicle"]
            self.term = self.base.command_manager.get_term("base_velocity")
            self.term.manual = True
            self.contact_sensor = self.base.scene["vehicle_contact"]
            matrix = self.contact_sensor.data.force_matrix_w
            if matrix is None or matrix.torch.shape[-2] == 0:
                raise RuntimeError("Filtered H1-vehicle contact sensor unavailable; refusing to train")
            self.origins = self.base.scene.env_origins
            self.goal_xy = self.origins[:, :2] + self.origins.new_tensor(self.layout.goal_position[:2])
            self.tracker = CrossingRewardTracker(num_envs, rewards, device=device)
            from crossing_command_guard import CrossingCommandGuard
            self.command_guard = CrossingCommandGuard(num_envs, device, settings)
            self.traffic = BatchedTraffic(num_envs, device, seed+1000, config=traffic_config)
            if traffic_config.vehicle_palette == "multicolor":
                from vehicle_colors import VehicleAppearance
                self.appearance = VehicleAppearance(num_envs, seed)
            self.elapsed = torch.zeros(num_envs, device=device)
            self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
            self.max_episode_length = round(settings.episode_seconds / .1)
            self.vehicle_y = torch.full((num_envs,), self.layout.vehicle_start_position[1], device=device)
            self.pose = self.vehicle.data.root_pose_w.torch.clone()
            if perception_config is not None:
                from h1_rgbd import RGBDPerception
                self.perception = RGBDPerception(self.robot,
                    [self.base.scene[name+"_rgbd"] for name in perception_config.camera_names],
                    self.origins, perception_config, perception_debug, headless,
                    road_length=self.layout.road_length, road_width=self.layout.road_width)
            if radar_config is not None:
                from radar_rig import create_radar
                if num_envs == 1:
                    self.radar = create_radar(self.robot, radar_config, radar_output, headless=headless,
                        observation_lane_y=float(self.origins[0, 1])+self.layout.vehicle_start_position[1]
                        if radar_observations else None)
                else:
                    if not radar_observations:
                        raise ValueError("Multi-env radar is supported in camera_radar observation mode only")
                    from radar_batch import H1RadarBatch
                    self.radar = H1RadarBatch(self.robot, radar_config, radar_output,
                        (self.origins[:, 1]+self.layout.vehicle_start_position[1]).cpu().tolist(), headless=headless)
            self.reset()
            if not headless:
                origin = self.origins[0].cpu().tolist()
                eye = tuple(a+b for a, b in zip(self.layout.camera_eye, origin))
                target = tuple(a+b for a, b in zip(self.layout.camera_target, origin))
                if self.base.viewport_camera_controller is not None:
                    self.base.viewport_camera_controller.update_view_location(eye=eye, lookat=target)
                else:
                    self.base.sim.set_camera_view(eye=eye, target=target)
            print("[Behavior] Crossing rewards only; native locomotion reward/termination managers are not stepped.", flush=True)
        except Exception:
            if self.radar:
                self.radar.close()
            self.base.close()
            raise

    def place_vehicle(self):
        self.pose[:, :3] = self.origins
        self.pose[:, 0] += self.traffic.x
        self.pose[:, 1] += self.vehicle_y
        self.pose[:, 2] += self.layout.vehicle_start_position[2]
        self.pose[:, 3:] = self.pose.new_tensor(self.layout.vehicle_orientation)
        self.vehicle.write_root_pose_to_sim_index(root_pose=self.pose)
        velocity = torch.stack((self.traffic.speed, torch.zeros_like(self.traffic.speed)), dim=1)
        set_scripted_vehicle_velocity(self.base, velocity)

    def reset(self, ids=None):
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        self.base.reset(env_ids=ids.int())
        self.traffic.reset(ids)
        if self.appearance:
            self.appearance.on_pass(ids, self.traffic.pass_count)
        self.vehicle_y[ids] = self.layout.vehicle_start_position[1]
        self.place_vehicle()
        self.base.scene.write_data_to_sim()
        self.base.sim.forward()
        self.elapsed[ids] = 0
        self.episode_length_buf[ids] = 0
        self.tracker.reset(self.robot.data.root_pos_w.torch[:, :2], ids)
        self.command_guard.reset(ids)
        if self.perception:
            self.perception.reset(ids, self.sim_time)
        if self.radar:
            if self.num_envs == 1:
                self.radar.reset(self.sim_time)
            else:
                self.radar.reset(self.sim_time, ids)
        return self.get_observations()

    def get_observations(self, *, sensor_only=False):
        robot = self.robot.data
        truth23 = getattr(self, "groundtruth23", False)
        if truth23 or (getattr(self, "oracle_observations", False) and not sensor_only):
            from oracle_observations import oracle_inputs
            velocity = torch.stack((self.traffic.speed, torch.zeros_like(self.traffic.speed)), dim=1)
            raw, flags, direction = oracle_inputs(
                robot.root_pos_w.torch[:, :2], robot.root_lin_vel_w.torch[:, :2],
                self.vehicle.data.root_pos_w.torch[:, :2], velocity, self.goal_xy)
        elif self.perception:
            raw, flags = self.perception.observations(self.sim_time, robot.root_pos_w.torch[:, :2],
                robot.root_lin_vel_w.torch[:, :2], self.goal_xy)
            if self.radar_observations:
                from radar_observations import radar_observation
                raw, flags, direction = radar_observation(raw, flags, self.radar.latest, self.sim_time,
                    robot.root_pos_w.torch[:, :2], robot.root_lin_vel_w.torch[:, :2], self.goal_xy)
        else:
            raw = self.base.observation_manager.compute()["crossing"]
        obs = behavior_observation(raw, robot.root_lin_vel_w.torch[:, :2], robot.root_quat_w.torch,
            self.term.vel_command_b, robot.root_pos_w.torch[:, :2]-self.tracker.anchor_xy,
            self.tracker.stationary_elapsed_s, self.elapsed, self.settings.episode_seconds)
        if self.perception or truth23:
            obs = torch.cat((obs, flags), dim=1)
            if self.radar_observations or truth23:
                obs = torch.cat((obs, direction), dim=1)
        groups = {"behavior": obs}
        if getattr(self, "privileged_critic", False) and not sensor_only:
            from asymmetric_observations import critic_observation
            velocity = torch.stack((self.traffic.speed, torch.zeros_like(self.traffic.speed)), dim=1)
            groups["critic"] = critic_observation(obs, robot.root_pos_w.torch[:, :2],
                robot.root_lin_vel_w.torch[:, :2], self.vehicle.data.root_pos_w.torch[:, :2], velocity)
        return TensorDict(groups, batch_size=[self.num_envs])

    @torch.no_grad()
    def step(self, actions, *, manual_command=None):
        # Explicit manual preview bypasses the crossing actor, goal-heading
        # assist and track-validity stop. Autonomous/training calls are unchanged.
        if manual_command is not None:
            if actions is not None:
                raise ValueError("Provide behavior actions OR a manual command, not both")
            manual_command = torch.as_tensor(manual_command, device=self.device, dtype=torch.float32)
            if manual_command.shape != (self.num_envs, 3):
                raise ValueError("Manual command must be [num_envs, 3]: vx, vy, yaw_rate")
            manual_command = clamp_commands(manual_command, "E")
            speed = None
        else:
            speed = speed_from_action(actions, self.settings.action_deadband)
        reward = torch.zeros(self.num_envs, device=self.device)
        totals = {name: torch.zeros_like(reward) for name in ("goal", "safe_episode", "vehicle_collision", "fall", "stationary", "timeout", "progress", "out_of_bounds")}
        events = {name: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                  for name in ("success", "collision", "fall", "timeout", "out_of_bounds")}
        terminal_observation = torch.zeros((self.num_envs, self.observation_dim), device=self.device)
        contact_peak = torch.zeros_like(reward)
        guard_log = {}
        for _ in range(self.settings.behavior_decimation):
            active = ~self.tracker.finished
            robot = self.robot.data
            desired = (manual_command.clone() if manual_command is not None else
                       heading_command(robot.root_pos_w.torch[:, :2], self.goal_xy,
                                       robot.root_quat_w.torch, speed, self.settings.heading_gain))
            desired[~active] = 0
            if self.perception and manual_command is None:
                sensor_obs = self.get_observations()["behavior"]
                valid = sensor_obs[:, 19] > .5
                if not self.radar_observations:
                    valid = valid & (sensor_obs[:, 20] > .5)
                # Camera-radar mode exposes invalid velocity/TTC to the actor
                # without overriding its command. Missing camera position still stops.
                # The dual-loss guard owns dropout handling: one valid source
                # must not be stopped by this older position-only gate.
                if not (self.settings.guard_enabled and self.command_guard.uses_dual_loss):
                    desired[~valid] = 0
            if self.perception and manual_command is None and self.settings.guard_enabled:
                # Pass the actor proposal for conflict prediction, even when
                # the legacy position-valid gate has already requested stop.
                proposal = heading_command(robot.root_pos_w.torch[:, :2], self.goal_xy,
                    robot.root_quat_w.torch, speed, self.settings.heading_gain)
                desired, diagnostic = self.command_guard.apply(proposal, sensor_obs, self.base.step_dt)
                desired[~active] = 0
                for name, value in diagnostic.items():
                    if value.dtype == torch.bool:
                        guard_log[name] = guard_log.get(name, torch.zeros_like(value)) | (value & active)
                    else:
                        guard_log[name] = value.clone()
            self.term.target.copy_(clamp_commands(desired, "E"))
            self.base.command_manager.compute(dt=self.base.step_dt)
            if self.trajectory_recorder is not None:
                trajectory_target = self.term.target.clone()
                trajectory_applied = self.term.vel_command_b.clone()
            low_obs = self.base.observation_manager.compute_group("policy")
            low_action = self.locomotion(low_obs)
            low_action[~active] = 0
            finite(low_obs, low_action)
            self.base.action_manager.process_action(low_action)
            collision = torch.zeros_like(active)
            fallen = torch.zeros_like(active)
            for _ in range(self.cfg.decimation):
                respawned = self.traffic.advance(self.base.physics_dt,
                    h1_xy=self.robot.data.root_pos_w.torch[:, :2]-self.origins[:, :2],
                    vehicle_y=self.vehicle_y)
                if self.appearance and len(respawned):
                    self.appearance.on_pass(respawned, self.traffic.pass_count)
                if self.radar and len(respawned):
                    # Radar history must not span teleports. Camera identity is
                    # retired by image association/dropout, never by traffic GT.
                    if self.num_envs == 1:
                        self.radar.reset(self.sim_time)
                    else:
                        self.radar.reset(self.sim_time, respawned)
                self.place_vehicle()
                self.base._sim_step_counter += 1
                self.base.action_manager.apply_action()
                self.base.scene.write_data_to_sim()
                self.base.sim.step(render=False)
                self.base.scene.update(dt=self.base.physics_dt)
                self.physics_steps += 1
                self.sim_time = self.physics_steps*self.base.physics_dt
                if self.perception and self.perception.due(self.physics_steps, self.base.physics_dt):
                    self.perception.follow()
                    if self.radar:
                        self.radar.follow()
                        self.radar.before_render(self.sim_time)
                    self.base.sim.render()
                    self.perception.capture(self.sim_time)
                    if self.radar:
                        self.radar.capture(self.sim_time,
                            self.perception.frames[0] if self.num_envs == 1 else self.perception.frames,
                            self.perception.cfg.vehicle_size)
                forces = self.contact_sensor.data.force_matrix_w.torch
                finite(forces)
                force_norm = torch.linalg.vector_norm(forces, dim=-1).flatten(1).amax(1)
                contact_peak = torch.maximum(contact_peak, force_norm)
                collision |= force_norm > self.settings.contact_threshold_n
                q = self.robot.data.root_quat_w.torch
                up_z = 1-2*(q[:, 0].square()+q[:, 1].square())
                height = self.robot.data.root_pos_w.torch[:, 2]-self.origins[:, 2]
                fallen |= (height < self.settings.fall_height_m) | (up_z < self.settings.fall_up_z)
            self.base.episode_length_buf += active.long()
            self.base.common_step_counter += 1
            self.elapsed += active.float()*self.base.step_dt
            local = self.robot.data.root_pos_w.torch[:, :2]-self.origins[:, :2]
            out_of_bounds = (local[:, 0].abs() > 55) | (local[:, 1].abs() > 8)
            timeout = (self.elapsed >= self.settings.episode_seconds-1e-5) & ~out_of_bounds
            parts = self.tracker.step(self.robot.data.root_pos_w.torch[:, :2], self.goal_xy,
                fallen=fallen, vehicle_contact=collision, timeout=timeout, dt=self.base.step_dt,
                out_of_bounds=out_of_bounds)
            reward += parts["total"]
            for name in totals:
                totals[name] += parts[name]
            events["success"] |= parts["success"]
            events["collision"] |= active & collision
            events["fall"] |= active & fallen
            events["timeout"] |= parts["truncated"]
            events["out_of_bounds"] |= parts["out_of_bounds_done"]
            newly_done = active & self.tracker.finished
            if self.trajectory_recorder is not None:
                self.trajectory_recorder.substep(self, active, desired, trajectory_target,
                    trajectory_applied, parts, collision, fallen, contact_peak)
            if newly_done.any():
                terminal_observation[newly_done] = self.get_observations()["behavior"][newly_done]
            # Radar is rendered once on the RGB-D schedule. Extra 50 Hz GUI
            # renders advance the native sensor clock without a matching mount
            # update and bias Doppler. Other play modes retain their old cadence.
            if not self.headless and self.radar is None:
                self.base.sim.render()
        self.episode_length_buf += 1
        done = self.tracker.finished.clone()
        terminal_elapsed = self.elapsed.clone()
        if done.any():
            self.reset(done.nonzero(as_tuple=False).flatten())
        observations = self.get_observations()
        finite(observations, reward)
        # Finite-horizon task: natural timeout is an episodic end with a penalty.
        # Do NOT send RSL-RL time_outs (which bootstraps continuation value).
        return observations, reward, done.long(), {
            "reward_terms": totals, "events": events, "terminal_observation": terminal_observation,
            "terminal_elapsed": terminal_elapsed, "contact_force_peak_n": contact_peak, "finite_horizon": True,
            "command_guard": guard_log}

    def close(self):
        if self.radar:
            self.radar.close()
        if self.perception:
            self.perception.close()
        self.base.close()
