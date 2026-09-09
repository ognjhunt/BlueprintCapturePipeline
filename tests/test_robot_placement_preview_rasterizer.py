import numpy as np

from blueprint_pipeline.robot_placement_preview_rasterizer import rasterize


def render(triangles, colours):
    return rasterize(triangles=np.asarray(triangles,dtype=float),colours=np.asarray(colours,dtype=float),
        basis=np.eye(3),low=np.array([-1.,-1.]),high=np.array([1.,1.]),
        crop_low=np.array([-1.,-1.,-1.]),crop_high=np.array([1.,1.,1.]),image_size=(100,100))


def test_front_scene_occludes_robot_independent_of_draw_order():
    back=[[-.8,-.8,0],[.8,-.8,0],[0,.8,0]]
    front=[[-.8,-.8,.5],[.8,-.8,.5],[0,.8,.5]]
    a,depth=render([front,back],[[0,255,0],[255,0,0]])
    b,_=render([back,front],[[255,0,0],[0,255,0]])
    assert np.array_equal(a,b)
    assert a[50,50,1]>0 and a[50,50,0]==0
    assert depth[50,50]==.5


def test_distant_wall_is_clipped_without_hiding_local_robot():
    back=[[-.8,-.8,0],[.8,-.8,0],[0,.8,0]]
    outside=[[-.8,-.8,2],[.8,-.8,2],[0,.8,2]]
    a,depth=render([back,outside],[[255,0,0],[0,255,0]])
    assert a[50,50,0]>0 and a[50,50,1]==0
    assert depth[50,50]==0


def test_large_triangle_is_clipped_per_fragment_not_by_centroid():
    triangle=[[-.8,-.8,0],[.8,-.8,0],[0,.8,3]]
    a,depth=render([triangle],[[255,0,0]])
    assert np.isfinite(depth).any()
    assert depth[np.isfinite(depth)].max() <= 1.
    assert not np.isfinite(depth[15,50])
