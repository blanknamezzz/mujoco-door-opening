"""Known-geometry door-sweep constraint, using only an independent MjData.

The query covers door assembly versus static obstacles and the commanded base
box. It is a geometric planning constraint, not a certificate for arm/contact
dynamics. Live door state is never changed by the query.
"""
import math
import numpy as np
import mujoco
from mobile_door import BASE_POSES,scene_xml


class DoorSweepChecker:
    def __init__(self, scene, step_deg=1., penetration_tolerance_m=.001):
        if not math.isfinite(step_deg) or step_deg<=0 or step_deg>5:
            raise ValueError('Sweep angular resolution must be in (0,5] degrees.')
        if not math.isfinite(penetration_tolerance_m) or not 0<=penetration_tolerance_m<=.005:
            raise ValueError('Bounded nonnegative penetration tolerance required.')
        self.m=mujoco.MjModel.from_xml_string(scene_xml(scene))
        self.data=mujoco.MjData(self.m)
        self.step_deg=step_deg;self.tolerance=penetration_tolerance_m
        self.door_q=int(self.m.jnt_qposadr[self.m.joint('door_hinge').id])
        self.base_q=np.array([self.m.jnt_qposadr[self.m.joint(n).id]
                             for n in ('mobile_x','mobile_y','mobile_yaw')],int)
        door_body=self.m.body('door').id
        assembly=[]
        for g,body in enumerate(self.m.geom_bodyid):
            ancestor=int(body)
            while ancestor and ancestor!=door_body:ancestor=int(self.m.body_parentid[ancestor])
            if ancestor==door_body and self.m.geom_contype[g]:assembly.append(g)
        obstacles=[g for g in range(self.m.ngeom) if self.m.geom(g).name.startswith(
            ('pillar_','wall_','clutter_','barrel_','obstacle_')) and self.m.geom_contype[g]]
        self.assembly=np.array(assembly,int)
        self.environment=np.array([*obstacles,self.m.geom('base_box').id],int)
        self.pairs=[(int(a),int(b)) for a in self.assembly for b in self.environment
            if (self.m.geom_contype[a]&self.m.geom_conaffinity[b]) or
               (self.m.geom_contype[b]&self.m.geom_conaffinity[a])]

    def check(self, base_id):
        if base_id not in range(4):raise ValueError('Physical station id required.')
        self.data.qpos[:]=self.m.qpos0
        self.data.qpos[self.base_q]=BASE_POSES[base_id]
        sign=1 if base_id<2 else -1
        checks=0
        for angle in np.linspace(0.,90.,math.ceil(90./self.step_deg)+1):
            self.data.qpos[self.door_q]=sign*math.radians(float(angle))
            mujoco.mj_forward(self.m,self.data)
            for a,b in self.pairs:
                delta=self.data.geom_xpos[a]-self.data.geom_xpos[b]
                radius=self.m.geom_rbound[a]+self.m.geom_rbound[b]+.002
                if delta@delta>radius*radius:continue
                checks+=1
                distance=mujoco.mj_geomDistance(self.m,self.data,a,b,.01,None)
                if distance < -self.tolerance:
                    return dict(allowed=False,base_id=base_id,first_blocked_angle_deg=float(angle),
                        pair=self.m.geom(a).name+':'+self.m.geom(b).name,
                        signed_distance_m=float(distance),distance_checks=checks,
                        step_deg=self.step_deg,penetration_tolerance_m=self.tolerance,
                        scope='known-map door assembly/static obstacles and commanded base box')
        return dict(allowed=True,base_id=base_id,first_blocked_angle_deg=None,pair=None,
                    signed_distance_m=None,distance_checks=checks,step_deg=self.step_deg,
                    penetration_tolerance_m=self.tolerance,
                    scope='known-map door assembly/static obstacles and commanded base box')

    def mask(self):
        results=[self.check(b) for b in range(4)]
        return np.array([r['allowed'] for r in results],bool),results
