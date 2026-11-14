"""Offline checks against false-positive success and privileged-policy shortcuts."""
import ast
import unittest
from types import SimpleNamespace as NS
import numpy as np
from cross_behavior.common import PROJECT, read
from cross_behavior.evaluator import Evaluator


class Model:
    def geom_id2name(self, i):
        return ['cubeA_g0','counter_top'][i]


def push_env(x=0.):
    data = NS(contact=[NS(geom1=0,geom2=1)], ncon=1,
              site_xpos=np.array([[0.,0.,1.]]), body_xpos=np.array([[x,0.,.52]]),
              body_xmat=np.eye(3).reshape(1,9),
              get_body_xvelp=lambda _:np.zeros(3), get_body_xvelr=lambda _:np.zeros(3))
    robot = NS(gripper={'right':NS(contact_geoms=['fingerL','fingerR'])}, eef_site_id={'right':0})
    return NS(sim=NS(data=data,model=Model()),robots=[robot],obj_body_id={'cubeA':0},
              objects={'cubeA':NS(root_body='cubeA',contact_geoms=['cubeA_g0'])},
              surface_z=.5,counter=NS(rot=0.),goal=np.array([0.,0.,.501]))


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.contract=read(PROJECT/'configs/tasks/cross_behavior_contracts.json')
        self.open=np.array([0.,0.,0.,0.,0.,0.,-1.])

    def evaluator(self):
        return Evaluator('push',self.contract,initial={'position':[-.1,0.,.52]})

    def test_transient_entry_does_not_pass(self):
        evaluator=self.evaluator()
        env=push_env()
        for t in range(1,20):
            self.assertFalse(evaluator.observe(env,self.open,t)['success'])
        self.assertTrue(evaluator.observe(env,self.open,20)['success'])

    def test_grasp_violation_cannot_be_erased_by_reopening(self):
        evaluator=self.evaluator()
        env=push_env()
        close=self.open.copy();close[6]=1.
        evaluator.observe(env,close,1)
        for t in range(2,30):
            self.assertFalse(evaluator.observe(env,self.open,t)['success'])

    def test_center_only_entry_does_not_pass(self):
        evaluator=self.evaluator()
        env=push_env(.045)
        for t in range(1,30):
            self.assertFalse(evaluator.observe(env,self.open,t)['in_goal'])

    def test_expert_has_no_environment_or_privileged_imports(self):
        tree=ast.parse((PROJECT/'cross_behavior/expert.py').read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(x and ('tasks' in x or 'evaluator' in x or 'robocasa' in x) for x in imports))
        forbidden={'sim','env','contact','success','body_xpos','obj_body_id'}
        names={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
        self.assertFalse(names & forbidden)

    def test_rgbd_support_plane_rejects_larger_red_cabinet(self):
        from cross_behavior.sensing import colored_position
        y,x=np.indices((30,30))
        xyz=np.stack((x*.005,y*.005,np.full_like(x,.9,dtype=float)),-1)
        rgb=np.full((30,30,3),100,dtype=np.uint8)
        rgb[:10,:10]=[200,90,50];xyz[:10,:10,2]=1.4
        rgb[18:23,18:23]=[255,0,0];xyz[18:23,18:23,2]=.94
        s=NS(rgb=rgb,xyz=xyz,depth=np.ones((30,30)))
        p=colored_position(s,'red',counter_block=True)
        np.testing.assert_allclose(p,[.1,.1,.92],atol=.001)


if __name__=='__main__':
    unittest.main()
