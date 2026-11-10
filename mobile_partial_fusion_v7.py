"""Reached-depth fusion ablation without cross-frame duplicate support.

This is a small voxel/ray-consistency study, not an OctoMap implementation.
Only actual reached depth and the obstacle-free CAD prior enter the mapper.
"""
from dataclasses import replace
import numpy as np
from mobile_partial_map_v5 import PartialDoorMap,MapSettings,observed_free

PROFILES=('voxel_union','ray_consistent_union','tight_voxel_union')


def union_voxels(points,voxel_m):
    """One deterministic actual endpoint per voxel, independent of repeats."""
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError('Finite Nx3 endpoints required.')
    if not np.isfinite(voxel_m) or voxel_m<=0:raise ValueError('Positive voxel size required.')
    if not len(points):return points.copy()
    keys=np.floor(points/voxel_m).astype(np.int64)
    center=(keys+.5)*voxel_m;distance=((points-center)**2).sum(1)
    order=np.lexsort((points[:,2],points[:,1],points[:,0],distance,keys[:,2],keys[:,1],keys[:,0]))
    ordered=keys[order];first=np.r_[True,np.any(ordered[1:]!=ordered[:-1],axis=1)]
    return points[order[first]].copy()


def fused_endpoints(point_sets,frames,settings,carve=False):
    """Free evidence may remove an endpoint; unknown rays never clear it."""
    cloud=np.concatenate(point_sets) if point_sets else np.empty((0,3))
    original=len(cloud);removed=0
    if carve and len(cloud):
        free=observed_free(cloud,frames,settings.visibility_margin_m)
        removed=int(free.sum());cloud=cloud[~free]
    cloud=union_voxels(cloud,settings.voxel_m)
    return cloud,dict(raw_frame_endpoints=original,free_ray_endpoints_removed=removed,
                      unique_occupied_voxels=len(cloud))


class FusedPartialDoorMap(PartialDoorMap):
    def __init__(self,prior,profile='voxel_union'):
        if profile not in PROFILES:raise ValueError('Unknown fusion profile.')
        settings=MapSettings()
        if profile=='tight_voxel_union':settings=replace(settings,point_radius_m=.003)
        super().__init__(prior,settings);self.profile=profile

    def query(self):
        if not self.frames:raise ValueError('At least one reached frame required.')
        cloud,statistics=fused_endpoints(self.points,self.frames,self.settings,
                                         self.profile=='ray_consistent_union')
        original=self.points
        try:
            self.points=[cloud]
            result=super().query()
        finally:self.points=original
        result.update(fusion_profile=self.profile,fusion_statistics=statistics,
                      point_radius_m=self.settings.point_radius_m)
        result['assumptions']+=['one actual endpoint per spatial voxel across all reached frames',
            'free-ray pruning only for ray_consistent_union; no future or hidden geometry',
            'voxel support and ray consistency are heuristics, not safety certificates']
        return result
