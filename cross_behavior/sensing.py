"""Explicit sensor boundary. Never extracts dynamic task object poses for the expert."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Sensors:
    rgb: np.ndarray
    depth: np.ndarray
    xyz: np.ndarray
    tcp: np.ndarray
    rotation: np.ndarray
    pad_local: np.ndarray
    left_pad_local: np.ndarray
    sep_local: np.ndarray
    approach_local: np.ndarray
    origin_rotation: np.ndarray
    instruction: str


def observe(env, camera):
    from robosuite.utils.camera_utils import get_real_depth_map, get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
    rgb, depth_buffer = env.sim.render(width=320, height=180, camera_name=camera, depth=True)
    # MuJoCo's image buffers are bottom-up; calibration helper uses top-down CV axes.
    rgb = rgb[::-1].copy()
    depth = get_real_depth_map(env.sim, depth_buffer[::-1]).copy()
    K = get_camera_intrinsic_matrix(env.sim, camera, 180, 320)
    extrinsic = get_camera_extrinsic_matrix(env.sim, camera)
    yy, xx = np.indices(depth.shape)
    rays = np.stack((xx, yy, np.ones_like(xx)), -1) @ np.linalg.inv(K).T
    xyz = (rays * depth[..., None]) @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    robot = env.robots[0]
    site = robot.eef_site_id['right']
    tcp = env.sim.data.site_xpos[site].copy()
    rot = env.sim.data.site_xmat[site].reshape(3, 3).copy()
    gripper = robot.gripper['right']
    geoms = gripper.important_geoms
    left = np.mean([env.sim.data.geom_xpos[env.sim.model.geom_name2id(n)] for n in geoms['left_fingerpad']], axis=0)
    right = np.mean([env.sim.data.geom_xpos[env.sim.model.geom_name2id(n)] for n in geoms['right_fingerpad']], axis=0)
    midpoint = (left + right) / 2
    root = env.sim.data.body_xpos[env.sim.model.body_name2id(gripper.root_body)].copy()
    sep = rot.T @ (right - left)
    sep /= np.linalg.norm(sep)
    approach = rot.T @ (midpoint - root)
    approach -= sep * np.dot(approach, sep)
    approach /= np.linalg.norm(approach)
    arm = robot.part_controllers['right']
    return Sensors(rgb, depth, xyz, tcp, rot, rot.T @ (midpoint - tcp),
                   rot.T @ (left - tcp), sep, approach, arm.origin_ori.copy(),
                   env.get_ep_meta()['lang'])


def colored_position(sensors, color, counter_block=False):
    """RGB threshold + metric depth only; no segmentation IDs, simulator bounding boxes or object poses."""
    import cv2
    rgb = sensors.rgb.astype(float)
    index = dict(red=0, green=1, blue=2)[color]
    other = [i for i in range(3) if i != index]
    mask = ((rgb[..., index] > 55) & (rgb[..., index] > 1.8 * rgb[..., other[0]])
            & (rgb[..., index] > 1.8 * rgb[..., other[1]]) & (sensors.depth < 2.5))
    if counter_block:
        # Estimate the support plane from RGB-D surface normals, not fixture truth.
        cloud = sensors.xyz
        dx = cloud[1:-1, 2:] - cloud[1:-1, :-2]
        dy = cloud[2:, 1:-1] - cloud[:-2, 1:-1]
        normals = np.cross(dx, dy)
        norm = np.linalg.norm(normals, axis=-1)
        horizontal = abs(normals[..., 2])/np.maximum(norm, 1e-12) > .97
        heights = cloud[1:-1, 1:-1, 2]
        samples = heights[horizontal & (heights > .6) & (heights < 1.15)]
        if len(samples) < 100:
            return None
        hist, edges = np.histogram(samples, bins=np.arange(.6, 1.155, .005))
        j = int(np.argmax(hist))
        plane = float(np.median(samples[(samples >= edges[j]) & (samples < edges[j+1])]))
        mask &= (cloud[..., 2] > plane+.006) & (cloud[..., 2] < plane+.075)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'), 8)
    if count < 2:
        return None
    candidates = list(range(1, count))
    if counter_block:
        candidates = [i for i in candidates if stats[i, cv2.CC_STAT_AREA] >= 4 and
                      max(np.ptp(sensors.xyz[labels == i][:, :2], axis=0)) < .10]
        if not candidates:
            return None
    component = max(candidates, key=lambda i:stats[i, cv2.CC_STAT_AREA])
    pts = sensors.xyz[labels == component]
    if len(pts) < 4:
        return None
    if color == 'blue':
        return np.median(pts, axis=0)
    # Upper visible face gives height; robust central XY avoids side-wall depth bias.
    top = np.quantile(pts[:, 2], .85)
    pts_top = pts[pts[:, 2] > top - .004]
    p = np.median(pts_top, axis=0)
    p[2] = top - (.025 if color == 'green' else .02)
    return p


def drawer_handle(sensors):
    """Conservative RGB-D geometric detection. Fail rather than using hidden handle state."""
    import cv2
    cloud = sensors.xyz[::2, ::2].reshape(-1, 3)
    cloud = cloud[(cloud[:, 2] > .45) & (cloud[:, 2] < 1.1) & (np.linalg.norm(cloud - sensors.tcp, axis=1) < 1.2)]
    if len(cloud) < 100:
        return None
    rng = np.random.default_rng(17)  # Deterministic perception, no outcome-based sampling.
    best = None
    for _ in range(150):
        a, b = cloud[rng.choice(len(cloud), 2, replace=False)]
        # Native cabinet fronts are vertical: estimate a vertical plane from
        # observed XY, rather than fitting a spurious plane tilted through trim.
        normal = np.array([b[1]-a[1], a[0]-b[0], 0.])
        length = np.linalg.norm(normal)
        if length < 1e-7:
            continue
        normal /= length
        if abs(normal[2]) > .2:
            continue
        score = np.sum(np.abs((cloud-a) @ normal) < .004)
        if best is None or score > best[0]:
            best = (int(score), a, normal)
    if best is None or best[0] < 100:
        return None
    _, center, normal = best
    if np.dot(sensors.tcp-center, normal) < 0:
        normal = -normal
    distance = (sensors.xyz-center) @ normal
    mask = ((distance > .008) & (distance < .065) & (sensors.xyz[..., 2] > .55)
            & (sensors.xyz[..., 2] < 1.05) & (sensors.depth < 2.5))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'), 8)
    candidates = []
    tangent = np.cross([0., 0., 1.], normal)
    tangent /= np.linalg.norm(tangent)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < 4:
            continue
        pts = sensors.xyz[labels == i]
        width = np.ptp(pts @ tangent)
        height = np.ptp(pts[:, 2])
        if .07 < width < .40 and height < .06 and width > 2*height:
            from scipy.optimize import least_squares
            # Source BarHandle CAD radius (handles.py), not dynamic task state.
            # Fit the visible cylindrical surface to its center, rather than
            # treating the visible upper/front surface as the grasp center.
            coordinates = np.column_stack((pts @ normal, pts[:,2]))
            radius = .013
            center0 = np.array([np.median(coordinates[:,0])-radius/2,
                                np.quantile(coordinates[:,1],.9)-radius])
            fit = least_squares(lambda center:np.linalg.norm(coordinates-center,axis=1)-radius,
                                center0, loss='soft_l1', f_scale=.001,
                                bounds=(coordinates.min(0)-radius,coordinates.max(0)+radius))
            if np.sqrt(np.mean(fit.fun**2)) > .005:
                continue
            point = np.median(pts,axis=0)
            point += normal*(fit.x[0]-np.dot(point,normal))
            point[2] = fit.x[1]
            candidates.append(point)
    if not candidates:
        return None
    # Native task explicitly identifies the top drawer. Highest visible horizontal handle.
    return max(candidates, key=lambda p: p[2]), normal
