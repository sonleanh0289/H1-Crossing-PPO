"""Procedural appearance only. Never imported by the image detector."""
PALETTE = {
    "green": (0.05, 0.55, 0.20),
    "red": (0.75, 0.04, 0.03),
    "blue": (0.03, 0.12, 0.80),
    "yellow": (0.85, 0.65, 0.02),
    "orange": (0.85, 0.22, 0.02),
    "purple": (0.50, 0.03, 0.70),
    "cyan": (0.02, 0.65, 0.75),
}


def color_for_pass(seed, env_id, pass_count):
    """Balanced cyclic coverage; independent of traffic RNG and policy actions."""
    if env_id < 0 or pass_count < 1:
        raise ValueError("Expected nonnegative env_id and positive pass_count")
    names = tuple(PALETTE)
    return names[(int(seed) + 3*int(env_id) + int(pass_count)-1) % len(names)]


class VehicleAppearance:
    """Per-environment materials; body/cabin only, no changes to physics/wheels."""
    def __init__(self, num_envs, seed):
        from isaaclab.sim import PreviewSurfaceCfg, get_current_stage
        from pxr import UsdShade
        self.seed = seed
        self.names = [None]*num_envs
        self.inputs = []
        stage = get_current_stage()
        for i in range(num_envs):
            inputs = []
            for part, suffix, metallic in (("Body", "", .25), ("Cabin", "/Cabin", .15)):
                material_path = f"/World/VehicleColors/env_{i}/{part}"
                config = PreviewSurfaceCfg(diffuse_color=PALETTE["green"], metallic=metallic, roughness=.35)
                config.func(material_path, config)
                material = UsdShade.Material(stage.GetPrimAtPath(material_path))
                mesh_path = f"/World/envs/env_{i}/Vehicle{suffix}/geometry/mesh"
                mesh = stage.GetPrimAtPath(mesh_path)
                if not mesh.IsValid():
                    raise ValueError(f"Missing vehicle geometry: {mesh_path}")
                UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material)
                shader = UsdShade.Shader(stage.GetPrimAtPath(material_path+"/Shader"))
                inputs.append(shader.GetInput("diffuseColor"))
            self.inputs.append(inputs)

    def set_color(self, env_id, name):
        from pxr import Gf
        color = PALETTE[name]
        for shader_input in self.inputs[env_id]:
            shader_input.Set(Gf.Vec3f(*color))
        self.names[env_id] = name

    def on_pass(self, ids, pass_counts):
        for i, count in zip(ids.tolist(), pass_counts[ids].tolist()):
            self.set_color(i, color_for_pass(self.seed, i, count))
