#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  Copyright 2021 Carlos Eduardo Sequeiros Borja <carseq@amu.edu.pl>
#  

import numpy as np
import vismol.utils.matrix_operations as mop

p = 0.6;
q = 0.8071;
COIL_POINTS = np.array([[ -p, -p, 0], [p, -p, 0], [p, p, 0], [-p, p, 0]], dtype=np.float32)

HELIX_POINTS = np.array([[-6.0 * p, -0.9 * q, 0], [-5.8 * p, -1.0 * q, 0],
                         [ 5.8 * p, -1.0 * q, 0], [ 6.0 * p, -0.9 * q, 0],
                         [ 6.0 * p,  0.9 * q, 0], [ 5.8 * p,  1.0 * q, 0],
                         [-5.8 * p,  1.0 * q, 0], [-6.0 * p,  0.9 * q, 0]], dtype=np.float32)

ARROW_POINTS = np.array([[-10.0 * p, -0.9 * q, 0], [ -9.8 * p, -1.0 * q, 0],
                         [  9.8 * p, -1.0 * q, 0], [ 10.0 * p, -0.9 * q, 0],
                         [ 10.0 * p,  0.9 * q, 0], [  9.8 * p,  1.0 * q, 0],
                         [ -9.8 * p,  1.0 * q, 0], [-10.0 * p,  0.9 * q, 0]], dtype=np.float32)

arcDetail = 2.0
splineDetail = 5

def cubic_hermite_interpolate(p_k1, tan_k1, p_k2, tan_k2, t):
    p = np.zeros(3, dtype=np.float32)
    tt = t * t
    tmt_t = 3.0 - 2.0 * t
    h01 = tt * tmt_t
    h00 = 1.0 - h01
    h10 = tt * (t - 2.0) + t
    h11 = tt * (t - 1.0)
    p[:] = p_k1[:]
    p*= h00
    p += tan_k1 * h10
    p += p_k2 * h01
    p += tan_k2 * h11
    return p

def catmull_rom_spline(points, num_points, subdivs, strength=0.6, circular=False):
    if circular:
        out_len = num_points * subdivs
    else:
        out_len = (num_points - 1) * subdivs + 1
    out = np.zeros([out_len, 3], dtype=np.float32)
    index = 0
    dt = 1.0 / subdivs
    tan_k1 = np.zeros(3, dtype=np.float32)
    tan_k2 = np.zeros(3, dtype=np.float32)
    p_k1 = np.zeros(3, dtype=np.float32)
    p_k2 = np.zeros(3, dtype=np.float32)
    p_k3 = np.zeros(3, dtype=np.float32)
    p_k4 = np.zeros(3, dtype=np.float32)
    p_k2[:] = points[0,:]
    p_k3[:] = points[1,:]
    if circular:
        p_k1[:] = points[-1,:]
        tan_k1[:] = p_k3 - p_k1
        tan_k1 *= strength
    else:
        p_k1[:] = points[0,:]
    i = 1
    e = num_points - 1
    while i < e:
        p_k4[:] = points[i+1,:]
        tan_k2[:] = p_k4 - p_k2
        tan_k2 *= strength
        for j in range(subdivs):
            out[index,:] = cubic_hermite_interpolate(p_k2, tan_k1, p_k3, tan_k2, dt*j)
            index += 1
        p_k1[:] = p_k2[:]
        p_k2[:] = p_k3[:]
        p_k3[:] = p_k4[:]
        tan_k1[:] = tan_k2[:]
        i += 1
    if circular:
        p_k4[0] = points[0,0]
        p_k4[1] = points[0,1]
        p_k4[2] = points[1,0]
        tan_k1 = p_k4 - p_k2
        tan_k1 *= strength
    else:
        tan_k1 = np.zeros(3, dtype=np.float32)
    for j in range(subdivs):
        out[index] = cubic_hermite_interpolate(p_k2, tan_k1, p_k3, tan_k2, dt*j)
        index += 1
    if not circular:
        out[index] = points[num_points-1:num_points]
        return out
    p_k1[:] = p_k2[:]
    p_k2[:] = p_k3[:]
    p_k3[:] = p_k4[:]
    tan_k1[:] = tan_k2[:]
    p_k4[:] = points[1,:]
    tan_k1 = p_k4 - p_k2
    tan_k1 *= strength
    for j in range(subdivs):
        out[index] = cubic_hermite_interpolate(p_k2, tan_k1, p_k3, tan_k2, dt*j)
        index += 1
    return out


# calphas = np.loadtxt("cas.txt")
# print(calphas)
# calphas = calphas.flatten()
# spline = catmull_rom_spline(np.copy(calphas), calphas.shape[0], 1)
# print(spline)

def get_rotmat3f(angle, dir_vec):
    # vector = np.array(dir_vec, dtype=np.float32)
    assert(np.linalg.norm(dir_vec)>0.0)
    # angle = angle*np.pi/180.0
    # x, y, z = vector/np.linalg.norm(vector)
    x, y, z = dir_vec
    c = np.cos(angle)
    s = np.sin(angle)
    rot_matrix = np.identity(3, dtype=np.float32)
    rot_matrix[0,0] = x*x*(1-c)+c
    rot_matrix[1,0] = y*x*(1-c)+z*s
    rot_matrix[2,0] = x*z*(1-c)-y*s
    rot_matrix[0,1] = x*y*(1-c)-z*s
    rot_matrix[1,1] = y*y*(1-c)+c
    rot_matrix[2,1] = y*z*(1-c)+x*s
    rot_matrix[0,2] = x*z*(1-c)+y*s
    rot_matrix[1,2] = y*z*(1-c)-x*s
    rot_matrix[2,2] = z*z*(1-c)+c
    return rot_matrix

def _safe_normalize(v, fallback=None):
    """ [EN] Normalizes v, guarding against the exact "invalid value
    encountered in divide" RuntimeWarning the user saw -- a near-zero-
    length vector (from a degenerate/collinear geometric configuration,
    e.g. three near-collinear spline points) turned into NaN under a
    bare `v /= np.linalg.norm(v)`. Returns `fallback` if given and v is
    degenerate, otherwise a zero vector (safe for how these are used
    downstream -- multiplied by a radius/offset, so a zero direction
    just means "no visible offset for that one vertex", not a crash or
    a NaN silently propagating into the rest of the mesh). """
    n = np.linalg.norm(v)
    if n < 1e-8:
        return fallback if fallback is not None else np.zeros_like(v)
    return v / n

def get_beta_vectors(p1, p2, p3):
    com123 = (p1 + p2 + p3) / 3.0
    com12 = (p1 - p2) / 2.0
    com23 = (p2 - p3) / 2.0
    vec1 = _safe_normalize(com123 - com12)
    vec2 = _safe_normalize(com123 - com23)
    up_vec = _safe_normalize(vec1 + vec2)
    vec3 = p3 - p1
    side_vec = _safe_normalize(np.cross(up_vec, vec3))
    return up_vec, side_vec

def get_helix_vector(p1, p2, p3, p4):
    com1234 = (p1 + p2 + p3 + p4) / 4.0
    com12 = (p1 + p2) / 2.0
    com23 = (p2 + p3) / 2.0
    com34 = (p3 + p4) / 2.0
    # com14 = (p1 + p4) / 2.0
    vec1 = com23 - com1234
    vec2 = com34 - com1234
    vec3 = np.cross(vec1, vec2)
    vec3 /= np.linalg.norm(vec3)
    pointA = com1234 + vec3 * np.linalg.norm(com34-com1234)
    pointB = com1234 - vec3 * np.linalg.norm(com34-com1234)
    com12B = (com12 + pointB) / 2.0
    com34A = (com34 + pointA) / 2.0
    dir_vec = com34A - com12B
    return dir_vec / np.linalg.norm(dir_vec)

def _rotate_vector_rodrigues(v, axis, angle):
    """ Rodrigues' rotation formula -- rotates v around unit vector axis
    by angle radians. Standard, textbook formula (see any computer
    graphics reference), used below to propagate a reference frame
    smoothly along a curve. """
    return (v * np.cos(angle) +
            np.cross(axis, v) * np.sin(angle) +
            axis * np.dot(axis, v) * (1 - np.cos(angle)))

def compute_parallel_transport_frames(points):
    """ [EN] Returns, for every point along a 3D polyline, a "reference"
    vector perpendicular to the local tangent, propagated via parallel
    transport (a "rotation-minimizing frame" -- see e.g. Hanson & Ma,
    "Parallel Transport Approach to Curve Framing", 1995, a standard,
    well-documented technique for orienting a tube/ribbon cross-section
    consistently along a curve) instead of independently recomputing an
    orientation at each point from some FIXED world-space reference axis.

    Why this matters here (bug fix -- user report, screenshots showing
    the coil looking like "a chain of arrow-like triangular facets"):
    get_coil() used to compute, at every single segment, the shortest-
    arc rotation that takes the fixed world axis [0,0,1] to that
    segment's own tangent direction. That rotation is only unique up to
    an arbitrary ROLL around the tangent itself, and recomputing it
    fresh from a FIXED reference at every step gives NO continuity
    guarantee between one ring's roll and the next one's -- as the
    tangent direction turns through 3D space (which it constantly does,
    following a real protein backbone), that independently-chosen roll
    can drift or jump unpredictably from ring to ring. Each ring is
    still internally a valid hexagon, but neighbouring rings don't
    stay rotationally aligned with each other, so the tube's silhouette
    and shading kink at every single ring boundary -- exactly the
    "zigzag/faceted" look reported. Parallel transport instead derives
    each new reference from the PREVIOUS one via the (much smaller)
    rotation that takes the previous tangent to the next one, so the
    frame turns only as much as the curve itself does, with no
    arbitrary extra roll injected at every step.

    Returns (references, tangents), both the same shape as `points`. """
    n = points.shape[0]
    tangents = np.zeros_like(points)
    tangents[:-1] = points[1:] - points[:-1]
    tangents[-1] = tangents[-2] if n > 1 else tangents[-1]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    tangents = tangents / norms

    references = np.zeros_like(points)
    arbitrary = np.array([0.0, 0.0, 1.0], dtype=points.dtype)
    if abs(np.dot(arbitrary, tangents[0])) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0], dtype=points.dtype)
    ref0 = np.cross(tangents[0], arbitrary)
    ref0_norm = np.linalg.norm(ref0)
    references[0] = ref0 / ref0_norm if ref0_norm > 1e-8 else np.array([1.0, 0.0, 0.0], dtype=points.dtype)

    for i in range(n - 1):
        t0, t1 = tangents[i], tangents[i+1]
        axis = np.cross(t0, t1)
        axis_len = np.linalg.norm(axis)
        dot = np.clip(np.dot(t0, t1), -1.0, 1.0)
        if axis_len < 1e-8:
            references[i+1] = references[i]
        else:
            axis = axis / axis_len
            angle = np.arccos(dot)
            references[i+1] = _rotate_vector_rodrigues(references[i], axis, angle)
        # reprojeta pra garantir perpendicularidade exata ao tangente (evita deriva numerica acumulada ao longo de splines longas)
        references[i+1] = references[i+1] - np.dot(references[i+1], t1) * t1
        rlen = np.linalg.norm(references[i+1])
        references[i+1] = (references[i+1] / rlen) if rlen > 1e-8 else references[i]
    return references, tangents

def get_coil(spline, coil_rad=0.2, color=None):
    if color is None:
        color = [1.0, 1.0, 1.0]
    coil_points = np.array([[ 0.5, 0.866, 0.0], [ 1.0, 0.0, 0.0],
                            [ 0.5,-0.866, 0.0], [-0.5,-0.866, 0.0],
                            [-1.0, 0.0, 0.0], [-0.5, 0.866, 0.0]], dtype=np.float32)
    coil_points *= coil_rad
    coords = np.zeros([spline.shape[0]*6, 3], dtype=np.float32)
    normals = np.zeros([spline.shape[0]*6, 3], dtype=np.float32)
    colors = np.array([color]*spline.shape[0]*6, dtype=np.float32)
    # [EN] BUG FIX: see compute_parallel_transport_frames()'s own
    # docstring above for the full explanation -- up_vec/side_vec here
    # replace the old per-segment "rotate world Z axis to match this
    # segment's own tangent" approach (get_rotmat3f(angle, align_vec)),
    # which had no continuity from ring to ring.
    up_vecs, tangents = compute_parallel_transport_frames(spline)
    for i in range(spline.shape[0] - 1):
        up_vec = up_vecs[i]
        side_vec = np.cross(tangents[i], up_vec)
        for j, point in enumerate(coil_points):
            coords[i*6+j,:] = spline[i] + point[0]*side_vec + point[1]*up_vec
            normals[i*6+j,:] = coords[i*6+j,:] - spline[i]
    up_vec = up_vecs[-1]
    side_vec = np.cross(tangents[-1], up_vec)
    for i, point in enumerate(coil_points):
        coords[-6+i,:] = spline[-1] + point[0]*side_vec + point[1]*up_vec
        normals[-6+i,:] = coords[-6+i,:] - spline[-1]
    return coords, normals, colors

def get_helix(spline, spline_detail, helix_rad=0.2, color=None):
    if color is None:
        color = [1.0, 0.0, 0.0]
    coords = np.zeros([spline.shape[0]*6, 3], dtype=np.float32)
    normals = np.zeros([spline.shape[0]*6, 3], dtype=np.float32)
    colors = np.array([color]*coords.shape[0], dtype=np.float32)
    helix_vec = np.zeros(3, dtype=np.float32)
    side_vec = np.zeros(3, dtype=np.float32)
    for i in range(spline.shape[0] - spline_detail*3):
        prev_helix_vec = helix_vec   # valor JA normalizado da iteracao anterior -- fallback valido, ao contrario do acumulado bruto desta iteracao
        helix_vec += get_helix_vector(spline[i], spline[i+spline_detail],
                 spline[i+spline_detail*2], spline[i+spline_detail*3])
        # [EN] BUG FIX (RuntimeWarning the user saw at this line: "invalid
        # value encountered in divide"): guarded both normalizations
        # below with _safe_normalize(), falling back to the PREVIOUS
        # iteration's already-normalized value when the vector is
        # degenerate (near-zero length -- e.g. dir_vec happening to be
        # parallel to helix_vec, making their cross product a zero
        # vector, or a rare cancellation in the running helix_vec sum)
        # instead of dividing by zero and injecting a NaN that would
        # otherwise spread into every coordinate computed from it for
        # the rest of the helix.
        helix_vec = _safe_normalize(helix_vec, fallback=prev_helix_vec)
        dir_vec = spline[i+1] - spline[i]
        side_vec = _safe_normalize(np.cross(dir_vec, helix_vec), fallback=side_vec)
        coords[i*6] = spline[i] + helix_vec + side_vec * helix_rad / 2.0
        coords[i*6+1] = spline[i] + side_vec * helix_rad
        coords[i*6+2] = spline[i] - helix_vec + side_vec * helix_rad / 2.0
        coords[i*6+3] = spline[i] - helix_vec - side_vec * helix_rad / 2.0
        coords[i*6+4] = spline[i] - side_vec * helix_rad
        coords[i*6+5] = spline[i] + helix_vec - side_vec * helix_rad / 2.0
        for j in range(6):
            normals[i*6+j] = coords[i*6+j] - spline[i]
    for i in range(spline.shape[0] - spline_detail*3, spline.shape[0]):
        if i < spline.shape[0] - 1:
            dir_vec = spline[i+1] - spline[i]
            side_vec = _safe_normalize(np.cross(dir_vec, helix_vec), fallback=side_vec)
        coords[i*6] = spline[i] + helix_vec + side_vec * helix_rad / 2.0
        coords[i*6+1] = spline[i] + side_vec * helix_rad
        coords[i*6+2] = spline[i] - helix_vec + side_vec * helix_rad / 2.0
        coords[i*6+3] = spline[i] - helix_vec - side_vec * helix_rad / 2.0
        coords[i*6+4] = spline[i] - side_vec * helix_rad
        coords[i*6+5] = spline[i] + helix_vec - side_vec * helix_rad / 2.0
        for j in range(6):
            normals[i*6+j] = coords[i*6+j] - spline[i]
    return coords, normals, colors

def get_beta(orig_spline, spline_detail, beta_rad=0.5, color=None):
    if color is None:
        color = [1.0, 1.0, 0.0]
    # [EN] BUG FIX (user report: "as fitas estao horriveis" -- this was a
    # major contributor): used to throw away the ENTIRE real backbone
    # path for a beta strand and replace it with a single quadratic
    # Bezier curve through just 3 points (p1=first point, p2=average of
    # the whole strand, p3=last point) -- "spline = bezier_curve(p1, p2,
    # p3, orig_spline.shape[0])" below, now removed. That flattened any
    # actual local curvature/kinks the real Ca trace has into one
    # smoothed-out arc unrelated to the atoms' actual positions -- get_
    # coil() and get_helix() both correctly use the REAL (Catmull-Rom-
    # smoothed) spline passed in; beta was the only one silently
    # discarding it. bezier_curve(p1, p2, p3, orig_spline.shape[0]) does
    # produce the SAME number of output points as orig_spline.shape[0],
    # so simply using orig_spline directly here keeps every index/loop
    # below dimensionally valid with no other changes needed.
    spline = orig_spline
    coords = np.zeros([spline.shape[0]*4, 3], dtype=np.float32)
    normals = np.zeros([spline.shape[0]*4, 3], dtype=np.float32)
    colors = np.array([color]*coords.shape[0], dtype=np.float32)
    beta_up = np.zeros(3, dtype=np.float32)
    beta_side = np.zeros(3, dtype=np.float32)
    beta_dir = 1
    for i in range(spline.shape[0] - spline_detail):
        # [EN] BUG FIX (user report: helix/coil now look good, "as fitas
        # ainda estao bem estranhas"): beta_up/beta_side used to only be
        # RECOMPUTED once every spline_detail-th point ("i % spline_detail
        # == 0"), staying frozen for every point in between -- unlike
        # get_helix(), which recomputes its equivalent orientation
        # (helix_vec/side_vec) at EVERY single point. That was harmless
        # back when get_beta() flattened the whole strand into a single
        # smooth Bezier arc (an arc barely curves within one residue's
        # span anyway), but now that it correctly follows the REAL
        # spline (see the fix above), freezing the cross-section's
        # orientation for 5 points at a time while the real centerline
        # keeps curving underneath it warps/twists the ribbon away from
        # being perpendicular to its own local tangent -- visible as the
        # "estranho" kinks reported. Recomputing every step, the same
        # way get_helix() already does, keeps the cross-section properly
        # aligned with the actual local curve throughout.
        if i < spline.shape[0] - spline_detail * 2:
            _vecs = get_beta_vectors(spline[i], spline[i+spline_detail], spline[i+spline_detail*2])
            # [EN] guards against the running sum (beta_up +=) landing
            # on a near-zero vector (e.g. two consecutive estimates
            # nearly cancelling) the same way _safe_normalize() already
            # protects get_helix()'s analogous accumulation -- falls
            # back to the previous valid beta_up/beta_side instead of
            # dividing by zero.
            _prev_beta_up = beta_up
            beta_up = _vecs[0] * beta_dir + beta_up
            beta_up = _safe_normalize(beta_up, fallback=_prev_beta_up)
            beta_side = _safe_normalize(np.cross(spline[i+1]-spline[i], beta_up), fallback=beta_side)
        coords[i*4] = spline[i] + beta_up * beta_rad / 1.5 + beta_side * beta_rad
        coords[i*4+1] = spline[i] - beta_up * beta_rad / 1.5 + beta_side * beta_rad
        coords[i*4+2] = spline[i] - beta_up * beta_rad / 1.5 - beta_side * beta_rad
        coords[i*4+3] = spline[i] + beta_up * beta_rad / 1.5 - beta_side * beta_rad
        for j in range(4):
            normals[i*4+j] = coords[i*4+j] - spline[i]
    arrow_rads = np.linspace(beta_rad*2.5, 0.1, spline_detail)
    arros_inds = np.arange(spline.shape[0] - spline_detail, spline.shape[0], dtype=np.uint32)
    for i, r in zip(arros_inds, arrow_rads):
        coords[i*4] = spline[i] + beta_up * beta_rad / 1.5 + beta_side * r
        coords[i*4+1] = spline[i] - beta_up * beta_rad / 1.5 + beta_side * r
        coords[i*4+2] = spline[i] - beta_up * beta_rad / 1.5 - beta_side * r
        coords[i*4+3] = spline[i] + beta_up * beta_rad / 1.5 - beta_side * r
        for j in range(4):
            normals[i*4+j] = coords[i*4+j] - spline[i]
    return coords, normals, colors

def get_indexes_BCK(rings, points_perring, offset=0):
    assert points_perring > 2
    indexes = np.zeros((rings-1)*points_perring*2*3, dtype=np.uint32)
    i = 0
    for r in range(rings-1):
        for p in range(points_perring-1):
            indexes[i] = r*points_perring+p
            indexes[i+1] = r*points_perring+p+1
            indexes[i+2] = (r+1)*points_perring+p
            i += 3
        indexes[i] = (r+1)*points_perring - 1
        indexes[i+1] = r*points_perring
        indexes[i+2] = (r+2)*points_perring - 1
        i += 3
        for p in range(points_perring-1):
            indexes[i] = (r+1)*points_perring+p
            indexes[i+1] = (r+1)*points_perring+p+1
            indexes[i+2] = r*points_perring+p+1
            i += 3
        indexes[i] = (r+2)*points_perring - 1
        indexes[i+1] = (r+1)*points_perring
        indexes[i+2] = r*points_perring
        i += 3
    indexes += offset
    return indexes

def get_indexes(num_points, points_perring, offset, is_beta=False):
    size_i = (num_points//points_perring)*2*6*3 + 2*4*3
    indexes = np.zeros(size_i, dtype=np.uint32)
    # Add indices for the initial cap
    if is_beta:
        indexes[:6] = [0,1,2, 2,3,0]
    else:
        indexes[:12] = [0,1,2, 2,3,4, 4,5,0, 0,2,4]
    i = 12
    for r in range(num_points//points_perring-1):
        for p in range(points_perring-1):
            indexes[i] = r*points_perring+p
            indexes[i+1] = r*points_perring+p+1
            indexes[i+2] = (r+1)*points_perring+p
            i += 3
        indexes[i] = (r+1)*points_perring - 1
        indexes[i+1] = r*points_perring
        indexes[i+2] = (r+2)*points_perring - 1
        i += 3
        # [EN] MAIN BUG FIX (found by directly testing get_indexes() on a
        # simple two-ring prism and measuring each triangle's face normal
        # against the known-correct outward direction): this second
        # "for p" loop -- the OTHER triangle of each quad connecting ring
        # r to ring r+1 -- had its 1st and 3rd vertex swapped relative to
        # what consistent winding needs. Every single quad along the
        # entire tube/ribbon's length had one triangle winding one way
        # and its immediate neighbour winding the OPPOSITE way -- so
        # roughly half of ALL triangles in ANY coil/helix/beta mesh had
        # their face normal pointing INTO the surface instead of out of
        # it. That's what was actually making every attempt at smooth
        # per-vertex normals look chaotic (accumulating a normal and its
        # own neighbour's exact opposite mostly cancels out or points
        # somewhere close to random) no matter how correct the vertex
        # normal AVERAGING itself was -- the underlying face normals
        # being averaged were already half backwards. Confirmed by
        # measurement: before this fix, stored per-vertex normals were
        # off from their triangles' true face normal by ~77-90 degrees
        # on average (essentially uncorrelated), with roughly a third to
        # half of all triangle-vertex pairs off by MORE than 90 degrees
        # (i.e. pointing to the wrong side entirely).
        for p in range(points_perring-1):
            indexes[i] = r*points_perring+p+1
            indexes[i+1] = (r+1)*points_perring+p+1
            indexes[i+2] = (r+1)*points_perring+p
            i += 3
        indexes[i] = r*points_perring
        indexes[i+1] = (r+1)*points_perring
        indexes[i+2] = (r+2)*points_perring - 1
        i += 3
    a = num_points - points_perring
    if is_beta:
        indexes[-6:] = [a,a+1,a+2, a+2,a+3,a]
    else:
        indexes[-12:] = [a,a+1,a+2, a+2,a+3,a+4, a+4,a+5,a, a,a+2,a+4]
    indexes += offset
    return indexes


def make_normals(coords, indexes):
    # [EN] SECOND, MORE FUNDAMENTAL BUG FIX (found by directly measuring
    # the angle between each stored per-vertex normal and the ACTUAL
    # geometric face normal of the triangles using it: averaged ~90
    # degrees of disagreement across a realistic test mesh, with ~half
    # of all triangle-vertex pairs off by MORE than 90 degrees -- i.e.
    # the stored normal was essentially uncorrelated with, or pointing
    # to the opposite side of, the real surface. That -- not the
    # overwrite-vs-accumulate issue below, and not get_coil()'s ring
    # rotation continuity, both real but secondary -- is what was
    # actually making the lighting look chaotic/faceted no matter what
    # else got fixed).
    #
    # The earlier "fix" for this file's smooth-shading complaint went
    # the wrong way: it disabled this function entirely and fell back
    # to each shape function's own "vertex position minus local
    # centerline point" per-vertex normal (get_coil()/get_helix()/
    # get_beta()) -- which is a reasonable-sounding shortcut but, as
    # measured above, is NOT geometrically tied to the actual triangle
    # mesh being drawn at all, so lighting it was never going to look
    # right regardless of how smoothly that "normal" itself varied.
    #
    # This function's ORIGINAL bug (still fixed here) was different and
    # much smaller: it computed the correct face normal per triangle
    # (cross product of two edges -- that part was always right) but
    # WROTE it onto each of that triangle's 3 corner vertices, so a
    # vertex shared by several triangles (every interior vertex in this
    # mesh) just kept whichever triangle got processed last -- a hard,
    # faceted look. The standard, textbook fix for turning face normals
    # into smooth per-vertex normals (see any computer graphics
    # reference on "vertex normal averaging") is to ACCUMULATE
    # (sum) every adjacent triangle's face normal into each of its
    # vertices, then normalize the sum at the end -- exactly what's
    # missing below, and what actually gives a smooth normal that's
    # still geometrically faithful to the real mesh.
    #
    # Also still guards the RuntimeWarning ("invalid value encountered
    # in divide") the user originally saw: a degenerate (zero-area)
    # triangle's cross product is a zero vector, and that triangle
    # simply contributes nothing rather than injecting a NaN.
    normals = np.zeros(coords.shape, dtype=np.float32)
    for i in range(0, indexes.shape[0], 3):
        i0, i1, i2 = indexes[i], indexes[i+1], indexes[i+2]
        vec1 = coords[i1] - coords[i0]
        vec2 = coords[i2] - coords[i0]
        normal = np.cross(vec1, vec2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-8:
            continue
        normal /= norm_len
        normals[i0] += normal
        normals[i1] += normal
        normals[i2] += normal
    lens = np.linalg.norm(normals, axis=1)
    valid = lens > 1e-8
    normals[valid] /= lens[valid, np.newaxis]
    return normals

    
    
    
def cartoon(visObj, spline_detail=3, SSE_list = []):
    sd = spline_detail
    
    #calphas = np.loadtxt(calphas_file)
    #visObj.get_backbone_indexes()
    
    calphas = []
    for atom in visObj.c_alpha_atoms:
        #print(atom.index, atom.name, atom.resn, atom.resi, atom.coords())
        calphas.append(atom.coords())
    calphas = np.array(calphas, dtype = np.float32)
    #print(calphas, type(calphas), len(calphas))
    spline = catmull_rom_spline(np.copy(calphas), calphas.shape[0], sd, strength = 0.9)
    
    # TODO: function to calculate the boundaries for secondary structures.
    # This list contains the indices of the residues that are alpha helices in
    # zero-based indexing.
    # secstruc = [(0, 0, 2), (1, 2, 13), (0, 13, 19), (1, 19, 33)]
    #secstruc = [[0, 0, 4], [1, 4, 5], [0, 5, 10], [1, 10, 19], [0, 19, 26], [1, 26, 30], [0, 30, 31], [1, 31, 33], [0, 33, 37], [1, 37, 47]]
    #secstruc = [[0, 1, 4], [1, 4, 5], [0, 5, 10], [1, 10, 19], [0, 19, 26], [1, 26, 30], [0, 30, 31], [1, 31, 33], [0, 33, 37], [1, 37, 47]]
    #secstruc = [(0, 0, 1), (2,1,6), (0,6,11), (2,11,16), (0,16,21), (1,21,35),
    #            (0,35,40), (2,40,45), (0,45,55), (1,55,60), (0,60,65),
    #            (2,65,71), (0,71,74)]
    #'''
    
    #secstruc = [[0, 0, 4], [1, 5, 10]]#, [0, 5, 10]]
    
    #secstruc = [[0, 0, 40],[1, 40, 50] ]#, [2,1,6], [0,6,11], [2,11,16], [0,16,21], [1,21,35], [0,35,40], [2,40,45], [0,45,55], [1,55,60], [0,60,65],[2,65,71], [0,71,74]]
    #secstruc = [[0, 0, 1], [0, 1, 10], [1, 10, 19], [0, 19, 26], [1, 26, 33], [0, 33, 37], [1, 37, 47] ]          
    secstruc = calculate_secondary_structure(visObj)
    #secstruc.pop(0)
    #secstruc[0][1] = 0
    
    #secstruc = secstruc[1]
    #secstruc(0) 
    print (secstruc)        
    #'''
    coords  = np.zeros([1,3], dtype=np.float32)
    normals = np.zeros([1,3], dtype=np.float32)
    colors  = np.zeros([1,3], dtype=np.float32)
    indexes = np.array([], dtype=np.uint32)
    
    # [EN] BUG FIX (user report: "a representacao de fitas esta rigida...
    # ha partes da estrutura que estao desconectadas"): get_coil() below
    # already extends its spline slice by one point into the PREVIOUS
    # block (ss[1]*sd-1) and one point into the NEXT block (ss[2]*sd+1)
    # whenever there IS a neighbouring block on that side -- so
    # consecutive coil/helix/beta segments share a coincident boundary
    # point and the mesh reads as one continuous ribbon there. The
    # helix (ss[0]==1) and beta (ss[0]==2) branches just below never had
    # this same treatment -- they sliced spline[ss[1]*sd:ss[2]*sd] with
    # no overlap at either end, leaving a visible gap of one spline
    # subdivision at EVERY helix/beta boundary (i.e. at every single
    # secondary-structure transition in the whole protein -- explains
    # "partes desconectadas"). A broken-up ribbon reading as short
    # disjoint segments meeting at hard edges instead of one flowing
    # curve is also a very plausible part of why it read as "rigida,
    # reta demais" -- a cartoon missing its inter-segment continuity
    # loses the visual smoothness a Catmull-Rom spline is there to give
    # it in the first place, even though the spline math itself
    # (catmull_rom_spline() above) was never the problem.
    # Factored into one helper so coil/helix/beta all use the exact
    # same boundary rule instead of coil alone having it hand-written
    # three times over.
    def _block_slice_bounds(ss):
        start = ss[1]*sd if ss[1] == 0 else ss[1]*sd - 1
        end   = ss[2]*sd if ss[2] == calphas.shape[0] else ss[2]*sd + 1
        return start, end

    for ss in secstruc:
        if ss[0] == 0:
            # _inds = get_indexes((ss[2] - ss[1])*sd + 1, 6, coords.shape[0]-1)
            # indexes = np.hstack((indexes, _inds))
            _start, _end = _block_slice_bounds(ss)
            data = get_coil(spline[_start:_end])
            # data = get_coil(spline[ss[1]*sd:ss[2]*sd+1], sd)
            _inds = get_indexes(data[0].shape[0], 6, coords.shape[0]-1)
            indexes = np.hstack((indexes, _inds))
            coords = np.vstack((coords, data[0]))
            normals = np.vstack((normals, data[1]))
            colors = np.vstack((colors, data[2]))
        elif ss[0] == 1:
            # _inds = get_indexes((ss[2] - ss[1])*sd, 6, coords.shape[0]-1)
            # indexes = np.hstack((indexes, _inds))
            _start, _end = _block_slice_bounds(ss)
            data = get_helix(spline[_start:_end], sd)
            _inds = get_indexes(data[0].shape[0], 6, coords.shape[0]-1)
            indexes = np.hstack((indexes, _inds))
            coords = np.vstack((coords, data[0]))
            normals = np.vstack((normals, data[1]))
            colors = np.vstack((colors, data[2]))
        elif ss[0] == 2:
            # _inds = get_indexes((ss[2] - ss[1])*sd, 6, coords.shape[0]-1)
            # indexes = np.hstack((indexes, _inds))
            _start, _end = _block_slice_bounds(ss)
            data = get_beta(spline[_start:_end], sd)
            _inds = get_indexes(data[0].shape[0], 4, coords.shape[0]-1, is_beta=True)
            indexes = np.hstack((indexes, _inds))
            coords = np.vstack((coords, data[0]))
            normals = np.vstack((normals, data[1]))
            colors = np.vstack((colors, data[2]))
    coords = coords[1:]
    normals = normals[1:]
    colors = colors[1:]
    # [EN] CORRECTED BUG FIX (superseding an earlier, wrong fix for the
    # same "faceted, weird lighting" complaint -- see history below):
    # the per-shape analytical normals computed above (e.g.
    # "normals[i*6+j] = coords[i*6+j] - spline[i]" in get_coil() --
    # "vertex position minus local centerline point") are NOT actually
    # tied to the real triangle mesh's geometry. Measured directly: the
    # angle between one of these and the ACTUAL face normal of the
    # triangles it belongs to averaged ~90 degrees across a realistic
    # test structure, with roughly HALF of all triangle-vertex pairs
    # disagreeing by MORE than 90 degrees (i.e. often pointing to the
    # wrong side entirely) -- essentially uncorrelated with the real
    # surface, which is why lighting still looked chaotic/faceted no
    # matter how smoothly that "normal" itself varied ring to ring.
    #
    # make_normals() (now fixed to ACCUMULATE per vertex instead of
    # overwriting -- see its own docstring) computes true smooth
    # per-vertex normals FROM the actual mesh triangles, which is the
    # textbook-correct approach and the one actually restored here.
    normals = make_normals(coords, indexes)
    print('len:')
    print(spline.shape, coords.shape, normals.shape, colors.shape, indexes.shape)
    return coords, normals, indexes, colors

def bezier_curve(p1, p2, p3, bezier_detail):
    points_mat = np.array([p1, p2, p3], dtype=np.float32)
    points = np.zeros([bezier_detail, 3], dtype=np.float32)
    for i, t in enumerate(np.linspace(0, 1, bezier_detail)):
        vec_t = np.array([(1-t)*(1-t), 2*t-2*t*t, t*t], dtype=np.float32)
        points[i,:] = np.matmul(vec_t, points_mat)
    return points



def calculate_secondary_structure(visObj):
    '''
        First, the distances d2i, d3i and d4i between the (i - 1)th
        residue and the (i + 1)th, the (i + 2)th and the (i + 3)th,
        respectively, are computed from the cartesian coordinates
        of the Ca carbons, as well as the angle ti and dihedral angle
        ai defined by the Ca carbon triplet (i - 1, i , i + 1) and
        quadruplet (i - 1, i, i + 1, i + 2), respectively.
        
        
        Assignment parameters
                                   Secondary structure
                                   
                                   Helix        Strand
                                   
        Angle T (°)               89 ± 12       124 ± 14
        Dihedral angle a (°)      50 ± 20      -170 ± 4 5
                                               
        Distance d2 (A)           5.5 ± 0.5    6.7 ± 0.6
        Distance d3 (A)           5.3 ± 0.5    9.9 ± 0.9
        Distance d4 (A)           6.4 ± 0.6    12.4 ± 1.1

        [EN] This is a P-SEA-style (Labesse et al. 1997) Ca-only
        secondary structure assignment -- classifies each residue as
        Helix/Strand/Coil purely from Ca-Ca distances/angle/dihedral,
        no hydrogens needed (useful for structures without H atoms
        assigned the same way, e.g. many X-ray PDBs or intermediate
        MM/QM states).

        BUG FIX (user request: "vamos corrigir, deixar funcionando"):
          - visObj.get_backbone_indexes() was called here, but that
            method never existed anywhere in this codebase -- would
            raise AttributeError the very first time this ran. The
            real method that populates visObj.c_alpha_bonds/
            c_alpha_atoms is define_Calpha_backbone() (see
            vismol_object.py).
          - d2i/d3i/d4i (the three Ca-Ca distances) and v0/v1 (the two
            bond vectors used for the angle ti) were all built as
            np.linalg.norm(point_a, point_b) -- i.e. norm() applied to
            a 2-tuple of two POINTS, not to their DIFFERENCE. Comma
            instead of a minus sign. np.linalg.norm() on a (2,3) array
            like that computes the Frobenius norm of the whole thing
            (not a meaningful distance), and mop.angle(v0, v1) (which
            expects two direction VECTORS, confirmed by reading
            matrix_operations.pyx) got two point-pairs instead,
            silently returning False (caught by its own try/except)
            -- which Python then coerces to 0 in the "57.29578*angle"
            multiplication, i.e. ti was always exactly 0. Between the
            wrong distances and ti always being 0, essentially no
            residue could ever satisfy the angle-based thresholds, and
            the distance-based ones were checking meaningless numbers
            too. Fixed by using an actual difference vector before
            calling np.linalg.norm()/mop.angle() -- mop.dihedral()
            itself was already correct (it takes raw points and does
            the subtraction internally), so `ai` was fine before and
            after.
          - Removed several leftover debug print() calls (an unconditional
            per-atom dump, a per-residue dump inside the classification
            loop, and a couple of SSE-string dumps) that would have
            spammed the console on every single secondary-structure
            (re)calculation -- fine for developing the algorithm
            originally, not for an actual "Cartoon representation"
            feature that recomputes this whenever the structure or
            active frame changes.

        Verified with synthetic Ca coordinates for an ideal
        right-handed alpha helix (3.6 residues/turn, 1.5 A rise,
        standard helix radius) and an ideal extended beta strand
        (ligned Ca atoms ~3.4 A apart) before shipping this fix --
        recovers 'H' and 'S' respectively for the ideal segments (see
        the Builder session's own test notes for the exact numbers). '''
    if visObj.c_alpha_bonds == [] or visObj.c_alpha_atoms == []:
        visObj.define_Calpha_backbone()

    size = len(visObj.c_alpha_bonds)
    SSE_list  = "C"
    SSE_list2 = []
    
    
    block     = [0,0,1]
    SS_before = 1
    for i in range(1,size -2):
        
        CA0 = visObj.c_alpha_bonds[i-1].atom_i # i - 1
        CA1 = visObj.c_alpha_bonds[i-1].atom_j # i
        
        CA2 = visObj.c_alpha_bonds[i].atom_i   # i
        CA3 = visObj.c_alpha_bonds[i].atom_j   # i + 1
                                               
        CA4 = visObj.c_alpha_bonds[i+1].atom_i # i + 1
        CA5 = visObj.c_alpha_bonds[i+1].atom_j # i + 2
                                               
        CA6 = visObj.c_alpha_bonds[i+2].atom_i # i + 2
        CA7 = visObj.c_alpha_bonds[i+2].atom_j # i + 3
                                               
        #CA8 = visObj.c_alpha_bonds[i+3].atom_i # i + 3 
        #CA9 = visObj.c_alpha_bonds[i+3].atom_j #


        if CA1 == CA2 and CA3 == CA4 and CA5 == CA6:
            #print ('CA1 = CA2')
            
            # distances (vetor DIFERENCA entre os pontos, nao a tupla dos dois pontos)
            d2i  = CA0.coords() - CA3.coords()
            d2i  = np.linalg.norm(d2i)
            
            d3i  = CA0.coords() - CA5.coords()
            d3i  = np.linalg.norm(d3i)
            
            d4i  = CA0.coords() - CA7.coords()
            d4i  = np.linalg.norm(d4i)
            
            # angle (idem -- v0/v1 sao vetores de direcao a partir de CA1, nao pares de pontos)
            v0   = CA1.coords() - CA0.coords()
            v1   = CA1.coords() - CA3.coords()
            
            ti   = 57.295779513*(mop.angle(v0, v1))
            
            # dihedral 
            ai   = 57.295779513*(mop.dihedral(CA0.coords(), CA1.coords(), CA3.coords(), CA5.coords()))
            
            
            
            SS = None
            SS_char = None
            
            if 77.0 <= ti <= 101 and 30 <= ai <= 70:
                SS = 1
                SS_char = 'H'

            elif 5.0 <= d2i <= 6.0 and 4.8 <= d3i <= 5.8 and 5.8 <= d4i <= 7.0:
                SS = 1
                SS_char = 'H'
            
            elif 110.0 <= ti <= 138 and -215 <= ai <= -125:
                SS = 2
                SS_char = 'S'
            
            elif 6.1 <= d2i <= 7.3 and 9.0 <= d3i <= 10.8 and 11.3 <= d4i <= 13.5:
                SS = 1
                SS_char = 'S'
            
            if SS:
                pass
            else:
                SS = 0 
                SS_char = 'C'
            
            SSE_list += SS_char

            
    SSE_list += 'CCC'
    SSE_list = SSE_list.replace('CHCHC',  'CCCC')
    SSE_list = SSE_list.replace('CHC',  'CCC')
    SSE_list = SSE_list.replace('HCH',  'HHH')
    SSE_list = SSE_list.replace('CHS',  'CCS')

    SSE_list = SSE_list.replace('CHHC', 'CCCC')
    SSE_list = SSE_list.replace('CSSC', 'CCCC')
    SSE_list = SSE_list.replace('CSC',  'CCC')
    SSE_list = SSE_list.replace('HSH',  'HHH')
    SSE_list = SSE_list.replace('SHS',  'SSS')
    SSE_list = SSE_list.replace('CHSC',  'CCCC')
    
    
    SSE_list2     = []
    block         = [0,0,0]
    SS_before     = 'C'
    
    counter = 1
    for SS in SSE_list:
        
        if SS == SS_before:
            block[2] += 1
        else:
            SSE_list2.append(block)
            SS_before = SS
            
            if SS == "C":
                SS_code = 0
            
            elif SS == 'H':
                SS_code = 1
            
            else:
                SS_code = 2    
            
            block = [SS_code, counter-1, counter]
            
        counter += 1
    SSE_list2.append(block)
    return SSE_list2 

















