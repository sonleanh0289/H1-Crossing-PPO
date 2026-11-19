"""Display-only H1 first-person viewport. No sensors or policy observations."""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class H1CameraConfig:
    body_name: str = "torso_link"
    offset: tuple[float, float, float] = (0.20, 0.0, 0.60)
    width: int = 640
    height: int = 480
    horizontal_fov_deg: float = 90.0
    clipping: tuple[float, float] = (0.05, 100.0)
    prim_path: str = "/World/H1FirstPersonCamera"


def camera_frame(position, quaternion_xyzw, offset):
    """Return eye, target and up in world space from a link pose (XYZW)."""
    values = (*position, *quaternion_xyzw, *offset)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Non-finite camera pose")
    norm = math.sqrt(sum(v*v for v in quaternion_xyzw))
    if norm < 1e-8:
        raise ValueError("Zero camera quaternion")
    x, y, z, w = (v / norm for v in quaternion_xyzw)
    def rotate(v):
        # v' = v + 2*w*(q_xyz cross v) + 2*(q_xyz cross (q_xyz cross v)).
        tx, ty, tz = 2*(y*v[2]-z*v[1]), 2*(z*v[0]-x*v[2]), 2*(x*v[1]-y*v[0])
        return (v[0]+w*tx+y*tz-z*ty, v[1]+w*ty+z*tx-x*tz, v[2]+w*tz+x*ty-y*tx)
    local = rotate(offset)
    eye = tuple(a+b for a, b in zip(position, local))
    forward = rotate((1., 0., 0.))
    return eye, tuple(a+b for a, b in zip(eye, forward)), rotate((0., 0., 1.))


class H1Camera:
    """Follow physics link data, not a possibly stale USD/Fabric parent transform."""
    def __init__(self, robot, stage, window, cfg=H1CameraConfig()):
        from pxr import UsdGeom, Sdf
        self.robot, self.cfg, self.window = robot, cfg, None
        names = list(robot.body_names)
        if names.count(cfg.body_name) != 1:
            raise ValueError(f"Expected one camera mount body {cfg.body_name!r}, found {names}")
        self.body_index = names.index(cfg.body_name)
        if not stage.GetPrimAtPath(cfg.prim_path).IsA(UsdGeom.Camera):
            raise ValueError(f"Scene camera is missing: {cfg.prim_path}")
        camera = UsdGeom.Camera.Define(stage, cfg.prim_path)
        camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
        aperture = 20.955
        camera.CreateHorizontalApertureAttr(aperture)
        camera.CreateVerticalApertureAttr(aperture * cfg.height / cfg.width)
        camera.CreateFocalLengthAttr(aperture / (2 * math.tan(math.radians(cfg.horizontal_fov_deg) / 2)))
        camera.CreateClippingRangeAttr(cfg.clipping)
        camera_xform = UsdGeom.Xformable(camera.GetPrim())
        camera_xform.ClearXformOpOrder()
        camera_xform.SetResetXformStack(True)
        self.transform = camera_xform.AddTransformOp()
        self.updates = 0
        self.last_frame = None
        self.update()
        self.window = window
        if self.window is None:
            raise RuntimeError("Kit could not create the H1 Camera viewport")
        self.viewport = self.window.viewport_api
        self.viewport.camera_path = Sdf.Path(cfg.prim_path)
        self.viewport.resolution = (cfg.width, cfg.height)
        self.viewport.resolution_scale = 1.0

    def update(self):
        from pxr import Gf, Sdf
        pose = self.robot.data.body_link_pose_w.torch[0, self.body_index].tolist()
        eye, target, up = camera_frame(pose[:3], pose[3:], self.cfg.offset)
        matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up)).GetInverse()
        self.transform.Set(matrix)
        if self.window is not None:
            # Keep this a robot view even if viewport navigation changes selection.
            self.viewport.camera_path = Sdf.Path(self.cfg.prim_path)
        self.last_frame = {"eye": eye, "target": target, "up": up, "mount_pose_xyzw": pose}
        self.updates += 1

    def report(self):
        return {"config": asdict(self.cfg), "updates": self.updates, "last_frame": self.last_frame,
                "viewport": viewport_details(self.viewport),
                "display_only": True}

    def close(self):
        if self.window is not None:
            self.window.destroy()
            self.window = None


def viewport_details(viewport):
    from pxr import UsdGeom, UsdRender
    camera_path = str(viewport.camera_path)
    stage = viewport.stage
    camera = stage.GetPrimAtPath(camera_path)
    product = UsdRender.Product(stage.GetPrimAtPath(viewport.render_product_path))
    return {"camera_path": camera_path, "render_product": viewport.render_product_path,
            "resolution": list(viewport.resolution), "fps": viewport.fps,
            "product_camera": [str(p) for p in product.GetCameraRel().GetTargets()] if product else [],
            "world_transform": str(UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(0)) if camera else None}


def capture_viewports(viewports, output, sim, app):
    """Capture explicit viewport handles; focus must never select the output camera."""
    import asyncio
    import time
    import omni.kit.renderer_capture
    from omni.kit.viewport.utility import capture_viewport_to_file
    import json
    for filename, viewport in viewports.items():
        if viewport is None:
            raise RuntimeError(f"No viewport available for {filename}")
    (output / "viewport_details.json").write_text(json.dumps({name: viewport_details(vp) for name, vp in viewports.items()}, indent=2), encoding="utf-8")
    for filename, viewport in viewports.items():
        path = output / filename
        # Kit's future acknowledges the render AOV, not successful disk output.
        # After resets a stale render resource can be acknowledged without a file.
        # Request a fresh frame again, with bounded retries, and verify disk output.
        for attempt in range(3):
            capture = capture_viewport_to_file(viewport, path.as_posix())
            future = asyncio.ensure_future(capture.wait_for_result(completion_frames=30))
            deadline = time.monotonic() + 15
            while not future.done() and time.monotonic() < deadline and app.is_running():
                sim.render()
            if not future.done():
                future.cancel()
                raise RuntimeError(f"Viewport capture timed out: {filename}")
            if not future.result():
                raise RuntimeError(f"Viewport capture returned no image: {filename}")
            omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
            deadline = time.monotonic() + 2
            while not path.is_file() and time.monotonic() < deadline and app.is_running():
                sim.render()
            if path.is_file() and path.stat().st_size > 0:
                break
            print(f"[Capture] No file for {filename}; fresh-frame attempt {attempt + 1}/3.", flush=True)
        else:
            raise RuntimeError(f"Viewport capture did not write an image: {path}")


def gpu_memory_snapshot():
    """Total device memory in MiB (all processes), not just PyTorch allocations."""
    import subprocess
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=5, check=True,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return {line.split(",")[0].strip(): int(line.split(",")[1])
                for line in result.stdout.strip().splitlines()}
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
