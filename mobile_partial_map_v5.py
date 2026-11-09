"""Door-sweep evidence from reached RGB-D frames and an obstacle-free CAD prior.

Unseen space stays unknown. No obstacle geometry, segmentation, scene layout,
candidate outcome or future camera frame enters a query. Navigation remains a
separate known-map service in this controlled opening-planner ablation.
"""
from dataclasses import dataclass,replace
import math
import numpy as np
import mujoco
from mobile_door import BASE_POSES,scene_xml
from mobile_sweep_v4 import DoorSweepChecker


@dataclass(frozen=True)
class MapSettings:
    pixel_stride:int=2
    depth_edge_abs_m:float=.03
    depth_edge_relative:float=.01
    voxel_m:float=.025
    point_radius_m:float=.015
    self_filter_m:float=.008
    visibility_margin_m:float=.015
    minimum_collision_voxels:int=2
    sweep_step_deg:float=1.
    coverage_step_deg:float=5.
    coverage_spacing_m:float=.10

    def __post_init__(self):
        if self.pixel_stride<1 or self.minimum_collision_voxels<1:
            raise ValueError('Positive pixel stride and hit count required.')
        values=(self.depth_edge_abs_m,self.depth_edge_relative,self.voxel_m,self.point_radius_m,self.self_filter_m,
                self.visibility_margin_m,self.sweep_step_deg,
                self.coverage_step_deg,self.coverage_spacing_m)
        if not all(math.isfinite(x) and x>0 for x in values):
            raise ValueError('Finite positive map resolutions required.')
        if self.sweep_step_deg>5 or self.coverage_step_deg>10:
            raise ValueError('Opening sweep resolution too coarse.')


def cad_prior(scene):
    """Explicitly discard all external-obstacle and layout information."""
    return replace(scene,obstacles=(),scene_id='known_door',description='',
                   wall_layout='unknown',expected_open_side=scene.handle_faces[0])


def geom_sdf(points,kind,size,center,rotation):
    """Signed Euclidean distance for the primitive CAD shapes in this task."""
    p=(np.asarray(points,float)-center)@np.asarray(rotation).reshape(3,3)
    size=np.asarray(size)
    if kind==mujoco.mjtGeom.mjGEOM_BOX:
        q=np.abs(p)-size
        return np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(-1),0)
    if kind==mujoco.mjtGeom.mjGEOM_SPHERE:return np.linalg.norm(p,axis=-1)-size[0]
    if kind==mujoco.mjtGeom.mjGEOM_CAPSULE:
        q=p.copy();q[:,2]-=np.clip(q[:,2],-size[1],size[1])
        return np.linalg.norm(q,axis=-1)-size[0]
    if kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
        q=np.column_stack((np.linalg.norm(p[:,:2],axis=-1)-size[0],np.abs(p[:,2])-size[1]))
        return np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(-1),0)
    raise ValueError('Unsupported CAD primitive: '+str(kind))


def backproject(frame,stride=2,edge_abs_m=.03,edge_relative=.01):
    depth=np.asarray(frame['depth_m']);K=np.asarray(frame['K']);T=np.asarray(frame['world_from_camera'])
    if depth.ndim!=2 or K.shape!=(3,3) or T.shape!=(4,4):raise ValueError('Calibrated depth frame required.')
    if stride<1 or not np.isfinite(K).all() or not np.isfinite(T).all() or min(K[0,0],K[1,1])<=0:
        raise ValueError('Invalid camera calibration or pixel stride.')
    v,u=np.mgrid[0:depth.shape[0]:stride,0:depth.shape[1]:stride]
    z=depth[v,u];valid=np.isfinite(z)&(z>.01)&(z<8.)
    # Rasterized silhouettes may contain blended-depth endpoints that belong
    # to neither surface. Keep only locally consistent3x3 depth neighborhoods.
    padded=np.pad(depth,1,mode='edge');lo=np.full_like(z,np.inf);hi=np.full_like(z,-np.inf)
    for dv in range(3):
        for du in range(3):
            neighbor=padded[v+dv,u+du];lo=np.minimum(lo,neighbor);hi=np.maximum(hi,neighbor)
    valid&=(hi-lo)<edge_abs_m+edge_relative*z
    u,v,z=u[valid],v[valid],z[valid]
    pc=np.column_stack(((u+.5-K[0,2])*z/K[0,0],-(v+.5-K[1,2])*z/K[1,1],-z))
    return pc@T[:3,:3].T+T[:3,3]


def observed_free(points,frames,margin=.015):
    """A sample is free only if an actual depth ray passes it with clearance."""
    points=np.asarray(points,float);free=np.zeros(len(points),bool)
    for frame in frames:
        T=np.asarray(frame['world_from_camera']);K=np.asarray(frame['K']);depth=np.asarray(frame['depth_m'])
        pc=(points-T[:3,3])@T[:3,:3];z=-pc[:,2]
        front=z>.01;safe=np.where(front,z,1.)
        u=np.rint(pc[:,0]*K[0,0]/safe+K[0,2]-.5).astype(int)
        v=np.rint(-pc[:,1]*K[1,1]/safe+K[1,2]-.5).astype(int)
        valid=front&(u>=0)&(u<depth.shape[1])&(v>=0)&(v<depth.shape[0])
        ids=np.flatnonzero(valid);hit=depth[v[ids],u[ids]]
        good=np.isfinite(hit)&(hit>.01)&(hit<8.)
        free[ids]|=good&(z[ids]<hit-margin)
    return free


class PartialDoorMap:
    def __init__(self,prior,settings=None):
        if prior.obstacles:raise ValueError('Only an obstacle-free CAD prior may enter the mapper.')
        self.settings=settings or MapSettings();self.prior=cad_prior(prior)
        self.m=mujoco.MjModel.from_xml_string(scene_xml(self.prior));self.d=mujoco.MjData(self.m)
        self.door_q=int(self.m.jnt_qposadr[self.m.joint('door_hinge').id])
        names=('door_hinge','lift_z','shoulder','elbow','wrist_yaw','left_slide','right_slide')
        self.state_q=np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in names],int)
        self.base_q=np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in ('mobile_x','mobile_y','mobile_yaw')],int)
        body=self.m.body('door').id;assembly=[]
        for g,b in enumerate(self.m.geom_bodyid):
            ancestor=int(b)
            while ancestor and ancestor!=body:ancestor=int(self.m.body_parentid[ancestor])
            if ancestor==body and self.m.geom_contype[g]:assembly.append(g)
        self.assembly=assembly
        self.visible_prior=[g for g in range(self.m.ngeom)
            if self.m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_PLANE and self.m.geom_rgba[g,3]>0]
        self.base_allowed,self.base_queries=DoorSweepChecker(self.prior,self.settings.sweep_step_deg).mask()
        self.points=[];self.frames=[];self.frame_stats=[];self._samples={}

    def set_state(self,state):
        state=np.asarray(state)
        if state.shape!=(19,) or not np.isfinite(state).all():raise ValueError('Finite reached19-D proprioception required.')
        self.d.qpos[:]=self.m.qpos0;self.d.qpos[self.state_q]=state[:7]
        self.d.qpos[self.base_q]=[state[14],state[15],math.atan2(state[16],state[17])]
        mujoco.mj_forward(self.m,self.d)

    def sdf(self,points,g):
        return geom_sdf(points,self.m.geom_type[g],self.m.geom_size[g],
                        self.d.geom_xpos[g],self.d.geom_xmat[g])

    def add_frame(self,frame,state):
        self.set_state(state);s=self.settings
        points=backproject(frame,s.pixel_stride,s.depth_edge_abs_m,s.depth_edge_relative);raw=len(points)
        # This region contains the known door's entire sweep. Floor endpoints
        # are removed; very low external obstacles are outside this experiment.
        valid=(np.linalg.norm(points[:,:2],axis=1)<1.5)&(points[:,2]>.04)&(points[:,2]<1.65)
        points=points[valid];known=np.zeros(len(points),bool)
        for g in self.visible_prior:known|=self.sdf(points,g)<s.self_filter_m
        points=points[~known]
        if len(points):
            _,ids=np.unique(np.floor(points/s.voxel_m).astype(np.int32),axis=0,return_index=True)
            points=points[np.sort(ids)]
        self.points.append(points);self.frames.append({k:np.asarray(frame[k]).copy() for k in ('depth_m','K','world_from_camera')})
        stats=dict(frame_index=len(self.frames)-1,raw_depth_points=raw,obstacle_voxels=len(points),
                   known_surface_points_removed=int(known.sum()),reached_proprio=np.asarray(state).tolist())
        self.frame_stats.append(stats);return stats

    def _shape_samples(self,g):
        s=self.m.geom_size[g];kind=self.m.geom_type[g]
        if kind==mujoco.mjtGeom.mjGEOM_BOX:extent=s
        elif kind==mujoco.mjtGeom.mjGEOM_SPHERE:extent=np.repeat(s[0],3)
        elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:extent=np.array([s[0],s[0],s[1]+s[0]])
        elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:extent=np.array([s[0],s[0],s[1]])
        else:raise ValueError('Unsupported swept primitive.')
        axes=[np.unique(np.r_[np.linspace(-e,e,max(2,math.ceil(2*e/self.settings.coverage_spacing_m)+1)),0.]) for e in extent]
        points=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
        return points[geom_sdf(points,kind,s,np.zeros(3),np.eye(3))<=1e-8]

    def sweep_samples(self,sign):
        if sign in self._samples:return self._samples[sign]
        pieces=[];local={g:self._shape_samples(g) for g in self.assembly}
        for angle in np.linspace(0,90,math.ceil(90/self.settings.coverage_step_deg)+1):
            self.d.qpos[self.door_q]=sign*math.radians(float(angle));mujoco.mj_forward(self.m,self.d)
            pieces.extend(local[g]@self.d.geom_xmat[g].reshape(3,3).T+self.d.geom_xpos[g] for g in self.assembly)
        samples=np.concatenate(pieces)
        _,ids=np.unique(np.floor(samples/.04).astype(np.int32),axis=0,return_index=True)
        self._samples[sign]=samples[np.sort(ids)];return self._samples[sign]

    def query(self):
        if not self.frames:raise ValueError('At least one actually reached frame required.')
        s=self.settings;cloud=np.concatenate(self.points);side_results={}
        for sign in (1,-1):
            minimum=.25;hits=0;first=None
            for angle in np.linspace(0,90,math.ceil(90/s.sweep_step_deg)+1):
                self.d.qpos[self.door_q]=sign*math.radians(float(angle));mujoco.mj_forward(self.m,self.d)
                inside=np.zeros(len(cloud),bool)
                for g in self.assembly:
                    nearby=np.linalg.norm(cloud-self.d.geom_xpos[g],axis=1)<self.m.geom_rbound[g]+.05
                    if nearby.any():
                        distances=self.sdf(cloud[nearby],g);minimum=min(minimum,float(distances.min()))
                        inside[np.flatnonzero(nearby)]|=distances<s.point_radius_m
                hits=max(hits,int(inside.sum()))
                if hits>=s.minimum_collision_voxels:first=float(angle);break
            samples=self.sweep_samples(sign);free=observed_free(samples,self.frames,s.visibility_margin_m)
            side_results[sign]=dict(observed_obstruction=first is not None,
                first_observed_blocked_angle_deg=first,maximum_hit_voxels=hits,
                sampled_free_fraction=float(free.mean()),minimum_observed_clearance_m=minimum,
                coverage_sample_count=len(samples))
        queries=[];features=[]
        for b in range(4):
            side=side_results[1 if b<2 else -1];known=not bool(self.base_allowed[b])
            blocked=known or side['observed_obstruction']
            status='known_base_collision' if known else 'observed_obstruction' if blocked else (
                'sampled_observed_free' if side['sampled_free_fraction']>=.95 else 'unknown')
            queries.append(dict(base_id=b,allowed=not blocked,status=status,known_base_collision=known,**side))
            features.append([float(known),float(side['observed_obstruction']),side['sampled_free_fraction'],
                             float(np.clip(side['minimum_observed_clearance_m']/.25,-1,1))])
        return dict(mask=[q['allowed'] for q in queries],features=features,queries=queries,
            frames=len(self.frames),obstacle_voxels=sum(len(p) for p in self.points),
            assumptions=['known rigid door/handle CAD and robot proprioception',
                'external obstacle surfaces only from actual depth endpoints',
                'unseen samples remain unknown; sampled free is not a collision certificate',
                'known-map navigation held fixed; only opening sweep map information is ablated',
                'floor removed below4cm;1degree sweep and inflated surface voxels; no moving-arm certificate'])
