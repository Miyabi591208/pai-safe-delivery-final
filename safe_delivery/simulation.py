"""Actual MuJoCo dynamics; no pose replay or direct state edits after reset."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import numpy as np
import mujoco
from .perception import COLORS, Classifier
from .planning import plan, limit_change

DT = .01
CONTROL_DT = .05
MAX_TIME = 40.
GOALS = np.array([[2.25, 2.1], [2.25, 0.], [2.25, -2.1]])
METHODS = ('direct', 'astar', 'smooth')


@dataclass
class Scene:
    seed: int
    label: int
    start: list
    obstacles: list
    mass: float
    light: float
    cargo_rgb: list


def make_scene(seed, label=None):
    rng = np.random.default_rng(seed)
    label = int(rng.integers(3)) if label is None else label
    obstacles = [[-.75+rng.uniform(-.16, .16), -.85+rng.uniform(-.18, .18), .31, .58],
                 [.65+rng.uniform(-.18, .18), .7+rng.uniform(-.18, .18), .35, .66],
                 [.55+rng.uniform(-.15, .15), -1.8+rng.uniform(-.10, .10), .33, .30]]
    return Scene(seed, label, [-2.25, float(rng.uniform(-2.3, -1.65))], obstacles,
                 float(rng.uniform(3.2, 5.2)), float(rng.uniform(.65, 1.05)),
                 np.clip(COLORS[label]*rng.uniform(.8, 1.1, 3), 0, 1).tolist())


def xml_for(scene):
    obs = ''.join(f'<geom name="shelf{i}" type="box" pos="{x} {y} .42" size="{sx} {sy} .42" rgba=".24 .32 .40 1"/>'
                  for i, (x, y, sx, sy) in enumerate(scene.obstacles))
    goals = ''.join(f'<geom name="bay{i}" type="box" pos="{x} {y} .006" size=".38 .38 .006" contype="0" conaffinity="0" rgba="{r} {g} {b} .65"/>'
                    for i, ((x, y), (r, g, b)) in enumerate(zip(GOALS, COLORS)))
    rgb = ' '.join(str(x) for x in scene.cargo_rgb)
    return f'''<mujoco model="AI cargo delivery">
      <compiler angle="radian"/>
      <option timestep="{DT}" integrator="implicitfast" gravity="0 0 -9.81"/>
      <visual><global offwidth="960" offheight="720"/><quality shadowsize="2048"/></visual>
      <asset><texture name="grid" type="2d" builtin="checker" rgb1=".84 .87 .88" rgb2=".76 .80 .82" width="256" height="256"/>
      <material name="floor" texture="grid" texrepeat="6 6" reflectance=".05"/></asset>
      <worldbody>
        <light pos="-1 -2 6" dir="0 0 -1" diffuse="{scene.light} {scene.light} {scene.light}" ambient=".2 .2 .2"/>
        <geom name="floor" type="plane" size="3.3 3.3 .1" material="floor"/>
        {obs}{goals}
        <body name="robot" pos="0 0 .15">
          <joint name="x" type="slide" axis="1 0 0" damping="3"/>
          <joint name="y" type="slide" axis="0 1 0" damping="3"/>
          <geom name="base" type="cylinder" size=".23 .13" mass="{scene.mass}" rgba=".12 .21 .30 1" friction=".5 .01 .001"/>
          <geom name="cargo" type="box" pos="0 0 .26" size=".145 .145 .13" mass=".3" rgba="{rgb} 1"/>
          <geom name="front" type="box" pos=".18 0 .14" size=".03 .09 .02" mass=".01" contype="0" conaffinity="0" rgba=".9 .9 .9 1"/>
          <camera name="cargo_camera" pos="0 0 1.05" fovy="45"/>
        </body>
      </worldbody>
      <actuator><motor joint="x" gear="1" ctrllimited="true" ctrlrange="-15 15"/>
        <motor joint="y" gear="1" ctrllimited="true" ctrlrange="-15 15"/></actuator>
    </mujoco>'''


def setup(scene):
    m = mujoco.MjModel.from_xml_string(xml_for(scene))
    d = mujoco.MjData(m)
    d.qpos[:] = scene.start
    mujoco.mj_forward(m, d)
    return m, d


def observe(m, d):
    with mujoco.Renderer(m, height=160, width=160) as renderer:
        renderer.update_scene(d, camera='cargo_camera')
        return renderer.render().copy()


def run(scene, method, classifier, frame_callback=None):
    if method not in METHODS:
        raise ValueError(method)
    m, d = setup(scene)
    camera_image = observe(m, d)
    perception = classifier.predict(camera_image)
    inferred = perception['label']
    goal = GOALS[inferred] if inferred is not None else None
    route = None if goal is None else (np.array([scene.start, goal]) if method == 'direct'
                                      else plan(scene.start, goal, scene.obstacles))
    if route is None:
        return {'seed': scene.seed, 'method': method, 'success': False, 'collision': False,
                'reason': 'perception_rejected' if inferred is None else 'no_route',
                'perception': perception, 'sim_time': 0., 'path_m': 0.}, None
    waypoint, command = 1, np.zeros(2)
    positions, velocities, commands, times = [], [], [], []
    collision, goal_reached = False, False
    collision_step = None
    obstacle_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f'shelf{i}') for i in range(len(scene.obstacles))}
    robot_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, k) for k in ('base', 'cargo')}
    for step in range(int(MAX_TIME/DT)):
        if step % 5 == 0:
            delta = route[waypoint]-d.qpos
            distance = np.linalg.norm(delta)
            if distance < .09 and waypoint < len(route)-1:
                waypoint += 1
                delta = route[waypoint]-d.qpos
                distance = np.linalg.norm(delta)
            speed = min(.6, 1.6*distance)
            desired = delta/max(distance, 1e-8)*speed
            command = limit_change(command, desired, .55*CONTROL_DT) if method == 'smooth' else desired
            positions.append(d.qpos.copy())
            velocities.append(d.qvel.copy())
            commands.append(command.copy())
            times.append(float(d.time))
        # Velocity servo with damping feed-forward. Pose is advanced only by mj_step.
        d.ctrl[:] = np.clip(18*(command-d.qvel)+3*d.qvel, -15, 15)
        mujoco.mj_step(m, d)
        for c in d.contact[:d.ncon]:
            if ((int(c.geom1) in robot_ids and int(c.geom2) in obstacle_ids) or
                    (int(c.geom2) in robot_ids and int(c.geom1) in obstacle_ids)):
                collision = True
                collision_step = step if collision_step is None else collision_step
        if frame_callback is not None and step % 5 == 0:
            frame_callback(m, d, route, perception, collision, camera_image)
        if np.linalg.norm(d.qpos-goal) < .12 and np.linalg.norm(d.qvel) < .10:
            goal_reached = True
            break
        # A collision ends the baseline episode; damage is not rewarded.
        if collision:
            break
    positions, velocities, commands, times = map(np.array, (positions, velocities, commands, times))
    acc = np.diff(velocities, axis=0)/CONTROL_DT
    jerk = np.diff(acc, axis=0)/CONTROL_DT
    result = {'seed': scene.seed, 'method': method,
              'success': bool(goal_reached and not collision and inferred == scene.label),
              'goal_reached': goal_reached, 'collision': collision,
              'reason': 'collision' if collision else ('delivered' if goal_reached and inferred == scene.label else 'timeout_or_wrong_bay'),
              'perception': perception, 'true_label': scene.label,
              'sim_time': round(float(d.time), 3),
              'path_m': float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
              'accel_rms_m_s2': float(np.sqrt(np.mean(np.sum(acc*acc, axis=1)))) if len(acc) else 0.,
              'jerk_rms_m_s3': float(np.sqrt(np.mean(np.sum(jerk*jerk, axis=1)))) if len(jerk) else 0.,
              'command_accel_max': float(np.max(np.linalg.norm(np.diff(commands, axis=0), axis=1)/CONTROL_DT)) if len(commands)>1 else 0.,
              'final_xy': d.qpos.tolist(), 'route': route.tolist()}
    trace = {'time': times, 'xy': positions, 'velocity': velocities, 'command': commands}
    return result, trace


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}


def evaluate(split='dev'):
    out = Path('results') / split
    out.mkdir(parents=True, exist_ok=True)
    seeds = [10, 11, 12] if split == 'dev' else list(range(100, 130))
    scenes = [make_scene(s, label=i % 3) for i, s in enumerate(seeds)]
    manifest = {'split': split, 'seeds': seeds, 'scenes': [asdict(s) for s in scenes],
                'methods': list(METHODS), 'dt': DT, 'control_dt': CONTROL_DT,
                'max_time': MAX_TIME, 'source_sha256': source_hashes(),
                'model_sha256': hashlib.sha256(Path('models/color_mlp.npz').read_bytes()).hexdigest()}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    classifier = Classifier('models/color_mlp.npz')
    rows = []
    for s in scenes:
        for method in METHODS:
            result, trace = run(s, method, classifier)
            rows.append(result)
            if trace is not None:
                np.savez_compressed(out / f'{s.seed}_{method}.npz', **trace)
            print(s.seed, method, result['reason'], result['sim_time'], flush=True)
    (out / 'episodes.json').write_text(json.dumps(rows, indent=2)+'\n')
    summary = {}
    for method in METHODS:
        r = [x for x in rows if x['method'] == method]
        summary[method] = {'n': len(r), 'successes': sum(x['success'] for x in r),
                           'collisions': sum(x['collision'] for x in r)}
    paired = [s.seed for s in scenes if all(next(x for x in rows if x['seed']==s.seed and x['method']==m)['success'] for m in ('astar', 'smooth'))]
    for m in ('astar', 'smooth'):
        r = [x for x in rows if x['method']==m and x['seed'] in paired]
        summary[m]['paired_n'] = len(r)
        for field in ('sim_time', 'path_m', 'accel_rms_m_s2', 'jerk_rms_m_s3'):
            summary[m]['paired_mean_'+field] = float(np.mean([x[field] for x in r])) if r else None
    (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('dev', 'heldout'), default='dev')
    evaluate(parser.parse_args().split)
