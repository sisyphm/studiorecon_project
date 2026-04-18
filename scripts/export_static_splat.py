#!/usr/bin/env python3
"""
Export static snapshot (t=0) as standard formats for well-known viewers.
  - scene_t0.splat : antimatter15 format (32 bytes/gaussian). Works in:
      https://antimatter15.com/splat/
      @mkkellogg/gaussian-splatting-3d
      Spark
  - scene_t0.ply   : standard 3DGS PLY with full SH (degree 3). Works in:
      SuperSplat (https://superspl.at/view)
      Any viewer that accepts Inria 3DGS PLY

Contents: background + actors deformed at frame 0, all in world space.
"""
import numpy as np
import torch
import sys
import struct
import pickle
from pathlib import Path
from plyfile import PlyData, PlyElement

sys.path.insert(0, '/root/code/SparseRecon/ShowMak3r_RELEASE')
from showmak3r.pipeline.smpl_deform.deformer import SMPLDeformer
from showmak3r.pipeline.scene.gaussian_model import GaussianModel


def infer_sh(p):
    import math
    d = PlyData.read(str(p))
    e = [x.name for x in d.elements[0].properties if x.name.startswith('f_rest_')]
    return int(math.sqrt((len(e) + 3) / 3)) - 1


def cov6_to_scale_quat(cov6_tensor):
    """Batch eigendecomposition of 6-element covariance (xx,xy,xz,yy,yz,zz) -> scale + quaternion."""
    device = cov6_tensor.device
    N = cov6_tensor.shape[0]
    M = torch.zeros(N, 3, 3, device=device, dtype=torch.float32)
    M[:, 0, 0] = cov6_tensor[:, 0]; M[:, 0, 1] = cov6_tensor[:, 1]; M[:, 0, 2] = cov6_tensor[:, 2]
    M[:, 1, 0] = cov6_tensor[:, 1]; M[:, 1, 1] = cov6_tensor[:, 3]; M[:, 1, 2] = cov6_tensor[:, 4]
    M[:, 2, 0] = cov6_tensor[:, 2]; M[:, 2, 1] = cov6_tensor[:, 4]; M[:, 2, 2] = cov6_tensor[:, 5]
    ev, R = torch.linalg.eigh(M)
    ev = torch.clamp(ev, min=1e-12)
    s = torch.sqrt(ev)
    dets = torch.det(R)
    fix = torch.ones(N, 1, 1, device=device)
    fix[dets < 0] = -1
    R = R * fix
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = torch.zeros(N, 4, device=device)
    ss = torch.sqrt(torch.clamp(tr + 1, min=1e-8)) * 2
    q[:, 0] = 0.25 * ss
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / (ss + 1e-8)
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / (ss + 1e-8)
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / (ss + 1e-8)
    q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
    return s.cpu().numpy(), q.cpu().numpy()


def pack_splat(pos, scale, rot, rgb, opa):
    n = len(pos)
    buf = bytearray(n * 32)
    for i in range(n):
        o = i * 32
        struct.pack_into('fff', buf, o, *pos[i])
        struct.pack_into('fff', buf, o + 12, *scale[i])
        r = int(np.clip(rgb[i, 0] * 255, 0, 255))
        g = int(np.clip(rgb[i, 1] * 255, 0, 255))
        b = int(np.clip(rgb[i, 2] * 255, 0, 255))
        a = int(np.clip(opa[i] * 255, 0, 255))
        struct.pack_into('BBBB', buf, o + 24, r, g, b, a)
        q = np.clip(rot[i] * 128 + 128, 0, 255).astype(np.uint8)
        struct.pack_into('BBBB', buf, o + 28, *q)
    return bytes(buf)


def save_ply(out_path, xyz, normals, f_dc, f_rest, opa_raw, scale_log, rot, sh_degree):
    """Save standard Inria 3DGS PLY format.
    f_dc: (N, 3)  — DC SH term
    f_rest: (N, 3*((sh+1)^2 - 1))  — flattened rest SH (channel-major per 3DGS convention)
    opa_raw: (N,) raw (pre-sigmoid) opacity
    scale_log: (N, 3) log-scale
    rot: (N, 4) quaternion
    """
    n_rest = f_rest.shape[1]
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')]
    for i in range(n_rest):
        dtype.append((f'f_rest_{i}', 'f4'))
    dtype += [('opacity', 'f4'),
              ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
              ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')]
    arr = np.zeros(xyz.shape[0], dtype=dtype)
    arr['x'] = xyz[:, 0]; arr['y'] = xyz[:, 1]; arr['z'] = xyz[:, 2]
    arr['nx'] = normals[:, 0]; arr['ny'] = normals[:, 1]; arr['nz'] = normals[:, 2]
    arr['f_dc_0'] = f_dc[:, 0]; arr['f_dc_1'] = f_dc[:, 1]; arr['f_dc_2'] = f_dc[:, 2]
    for i in range(n_rest):
        arr[f'f_rest_{i}'] = f_rest[:, i]
    arr['opacity'] = opa_raw
    arr['scale_0'] = scale_log[:, 0]; arr['scale_1'] = scale_log[:, 1]; arr['scale_2'] = scale_log[:, 2]
    arr['rot_0'] = rot[:, 0]; arr['rot_1'] = rot[:, 1]; arr['rot_2'] = rot[:, 2]; arr['rot_3'] = rot[:, 3]
    el = PlyElement.describe(arr, 'vertex')
    PlyData([el]).write(str(out_path))


def main():
    model_path = Path('results/eval/001_grappling_8view/001_grappling_8view_exp_8view_seqrefine_stage2')
    actor_base = Path('results/eval/001_grappling_8view/actor_seqrefine')
    output_dir = Path('/root/code/StudioRecon_project/assets/static')
    output_dir.mkdir(parents=True, exist_ok=True)
    iter_path = model_path / 'iteration_10000'
    device = 'cuda:0'
    FRAME_IDX = 0
    TOP_N = 800000   # subsample to this many gaussians (by opacity), None = keep all

    # ---- Background ----
    print("[1/3] Loading background...")
    bg_sh = infer_sh(str(iter_path / 'stage_pcd.ply'))
    bg_gs = GaussianModel(bg_sh)
    bg_gs.load_ply(str(iter_path / 'stage_pcd.ply'))
    print(f"  BG: {bg_gs.get_xyz.shape[0]:,} gaussians, SH deg {bg_sh}")

    bg_xyz = bg_gs.get_xyz.detach().cpu().numpy()
    bg_opa_raw = bg_gs._opacity.detach().cpu().numpy().squeeze(-1)  # pre-sigmoid
    bg_opa = bg_gs.get_opacity.squeeze().detach().cpu().numpy()
    bg_scale_log = bg_gs._scaling.detach().cpu().numpy()  # log-scale
    bg_scale = bg_gs.get_scaling.detach().cpu().numpy()
    bg_rot = bg_gs.get_rotation.detach().cpu().numpy()
    bg_rot = bg_rot / (np.linalg.norm(bg_rot, axis=1, keepdims=True) + 1e-8)
    bg_dc = bg_gs._features_dc.detach().cpu().numpy().squeeze(1)  # (N, 3)
    # f_rest in pipeline shape: (N, (sh+1)^2-1, 3). PLY expects channel-major flat: 3*(sh+1)^2-1
    bg_rest_raw = bg_gs._features_rest.detach().cpu().numpy()  # (N, K, 3)
    bg_rest_flat = bg_rest_raw.transpose(0, 2, 1).reshape(bg_rest_raw.shape[0], -1)  # (N, 3*K)

    # RGB for .splat from DC term
    C0 = 0.2820947917738781
    bg_rgb = np.clip(0.5 + C0 * bg_dc, 0, 1).astype(np.float32)

    # ---- Actors (deformed at frame 0) ----
    print("[2/3] Deforming actors at frame 0...")
    actor_rows = []  # tuples (xyz, rgb, opa, opa_raw, scale, scale_log, rot, dc, rest_flat)
    for ply in sorted(iter_path.glob('actor_*.ply')):
        pnum = ply.stem.split('_')[1]
        sh_deg = infer_sh(str(ply))
        gs = GaussianModel(sh_deg)
        gs.load_ply(str(ply))
        print(f"  Actor {pnum}: {gs.get_xyz.shape[0]:,} gaussians, SH deg {sh_deg}")

        smpl_data = pickle.load(open(actor_base / pnum / 'optimized.pkl', 'rb'))
        fnames = sorted(smpl_data.keys())
        params = []
        for fn in fnames:
            p = smpl_data[fn]['smpl_param']
            if isinstance(p, np.ndarray):
                p = torch.from_numpy(p).float()
            params.append(p)
        smpl_params = torch.cat(params, dim=0).cuda()
        beta = smpl_params[0, 76:86]
        deformer = SMPLDeformer(gender='neutral', beta=beta, smpl_scale=1.0)

        m3D = gs.get_xyz.detach().to(device)
        cov = gs.get_covariance(1.0).detach().to(device)
        sp = smpl_params[FRAME_IDX:FRAME_IDX + 1]
        with torch.no_grad():
            dp, dc_cov, _ = deformer.deform_gp(m3D, cov, sp, cond=dict(img_idx=FRAME_IDX))
        scale, quat = cov6_to_scale_quat(dc_cov)
        scale_log = np.log(np.maximum(scale, 1e-8))

        opa_raw = gs._opacity.detach().cpu().numpy().squeeze(-1)
        opa = gs.get_opacity.squeeze().detach().cpu().numpy()
        dc = gs._features_dc.detach().cpu().numpy().squeeze(1)
        rest_raw = gs._features_rest.detach().cpu().numpy()
        rest_flat = rest_raw.transpose(0, 2, 1).reshape(rest_raw.shape[0], -1)
        rgb = np.clip(0.5 + C0 * dc, 0, 1).astype(np.float32)
        xyz_world = dp.cpu().numpy()

        actor_rows.append(dict(
            xyz=xyz_world, rgb=rgb, opa=opa, opa_raw=opa_raw,
            scale=scale, scale_log=scale_log, rot=quat,
            dc=dc, rest_flat=rest_flat, sh_deg=sh_deg,
        ))

    # SH degree check — bg and actors need same SH degree for a single PLY
    sh_degrees = [bg_sh] + [a['sh_deg'] for a in actor_rows]
    target_sh = min(sh_degrees)
    if not all(d == target_sh for d in sh_degrees):
        print(f"  WARN: mixed SH degrees {sh_degrees}, truncating to {target_sh}")
    n_rest_target = 3 * ((target_sh + 1) ** 2 - 1)

    def trunc_rest(flat, deg_cur):
        # flat is channel-major (N, 3*K_cur). Reshape to (N, 3, K_cur), slice, flatten back.
        K_cur = (deg_cur + 1) ** 2 - 1
        K_new = (target_sh + 1) ** 2 - 1
        if K_cur == K_new:
            return flat
        arr = flat.reshape(-1, 3, K_cur)
        arr = arr[:, :, :K_new]
        return arr.reshape(-1, 3 * K_new)

    bg_rest_flat = trunc_rest(bg_rest_flat, bg_sh)
    for a in actor_rows:
        a['rest_flat'] = trunc_rest(a['rest_flat'], a['sh_deg'])

    # Concatenate
    all_xyz = np.concatenate([bg_xyz] + [a['xyz'] for a in actor_rows], axis=0).astype(np.float32)
    all_rgb = np.concatenate([bg_rgb] + [a['rgb'] for a in actor_rows], axis=0).astype(np.float32)
    all_opa = np.concatenate([bg_opa] + [a['opa'] for a in actor_rows], axis=0).astype(np.float32)
    all_opa_raw = np.concatenate([bg_opa_raw] + [a['opa_raw'] for a in actor_rows], axis=0).astype(np.float32)
    all_scale = np.concatenate([bg_scale] + [a['scale'] for a in actor_rows], axis=0).astype(np.float32)
    all_scale_log = np.concatenate([bg_scale_log] + [a['scale_log'] for a in actor_rows], axis=0).astype(np.float32)
    all_rot = np.concatenate([bg_rot] + [a['rot'] for a in actor_rows], axis=0).astype(np.float32)
    all_dc = np.concatenate([bg_dc] + [a['dc'] for a in actor_rows], axis=0).astype(np.float32)
    all_rest = np.concatenate([bg_rest_flat] + [a['rest_flat'] for a in actor_rows], axis=0).astype(np.float32)

    n_total = len(all_xyz)
    print(f"  Total: {n_total:,} gaussians (pre-filter)")

    # ---- Subsample top-N by opacity ----
    if TOP_N is not None and TOP_N < n_total:
        keep = np.argsort(-all_opa)[:TOP_N]
        all_xyz = all_xyz[keep]; all_rgb = all_rgb[keep]; all_opa = all_opa[keep]
        all_opa_raw = all_opa_raw[keep]; all_scale = all_scale[keep]; all_scale_log = all_scale_log[keep]
        all_rot = all_rot[keep]; all_dc = all_dc[keep]; all_rest = all_rest[keep]
        n_total = len(all_xyz)
        print(f"  After top-{TOP_N} subsample: {n_total:,} gaussians")

    # ---- Write .splat ----
    print("[3/3] Writing outputs...")
    splat_bytes = pack_splat(all_xyz, all_scale, all_rot, all_rgb, all_opa)
    with open(output_dir / 'scene_t0.splat', 'wb') as f:
        f.write(splat_bytes)
    print(f"  scene_t0.splat    : {len(splat_bytes) / 1e6:.1f}MB")

    # ---- Write gzipped version too ----
    import gzip
    gz_bytes = gzip.compress(splat_bytes, compresslevel=9)
    with open(output_dir / 'scene_t0.splat.gz', 'wb') as f:
        f.write(gz_bytes)
    print(f"  scene_t0.splat.gz : {len(gz_bytes) / 1e6:.1f}MB ({100 * len(gz_bytes) / len(splat_bytes):.1f}% of original)")

    # ---- Write .ply with full SH ----
    normals = np.zeros_like(all_xyz)
    save_ply(output_dir / 'scene_t0.ply',
             all_xyz, normals, all_dc, all_rest,
             all_opa_raw, all_scale_log, all_rot, target_sh)
    ply_size = (output_dir / 'scene_t0.ply').stat().st_size / 1e6
    print(f"  scene_t0.ply   : {ply_size:.1f}MB (SH degree {target_sh})")
    print(f"  Output: {output_dir}")


if __name__ == '__main__':
    main()
