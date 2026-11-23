"""Render two validated lavender trajectories with the original GLB textures."""
import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector
import numpy as np


def main(asset, experiment, output):
    output.mkdir(parents=True, exist_ok=True)
    stats=json.loads((experiment/"result.json").read_bytes())
    with np.load(experiment/"curves.npz",allow_pickle=False) as data:
        times=data["time_s"]
        stems=data["tip_m"]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene
    order=np.argsort([-r["frequency_hz"] for r in stats["modal_validation"][0]])
    for case, shift in enumerate((-.23,.23)):
        before=set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(asset))
        imported=[obj for obj in bpy.data.objects if obj not in before and obj.type=="MESH"]
        if len(imported)!=len(order):
            raise ValueError("GLB mesh structure differs from the saved modal model")
        world=[[obj.matrix_world@v.co for v in obj.data.vertices] for obj in imported]
        bounds=[(min(v.z for v in verts),max(v.z for v in verts)) for verts in world]
        rank=np.argsort([hi-lo for lo,hi in bounds])
        low,high=min(lo for lo,_ in bounds),max(hi for _,hi in bounds)
        scale=stats["height_m"]/(high-low)
        for ix, stem in zip(rank,order):
            obj=imported[ix]
            verts=world[ix]
            lo,hi=bounds[ix]
            obj.parent=None
            obj.matrix_world=Matrix.Identity(4)
            for v,p in zip(obj.data.vertices,verts):
                v.co=(p.x*scale+shift,p.y*scale,(p.z-low)*scale)
            base=obj.shape_key_add(name="Basis")
            bent=obj.shape_key_add(name="Tip displacement (m)")
            for dst,v,p in zip(bent.data,base.data,verts):
                u=(p.z-lo)/(hi-lo)
                dst.co=v.co+Vector((u*u*(6-4*u+u*u)/3,0,0))
            bent.slider_min=-1
            bent.slider_max=1
            for k,d in enumerate(stems[case,stem],1):
                bent.value=float(d)
                bent.keyframe_insert(data_path="value",frame=k)
    def object_at(kind,name,location):
        item=bpy.data.cameras.new(name) if kind=="CAMERA" else bpy.data.lights.new(name,type="AREA")
        obj=bpy.data.objects.new(name,item)
        scene.collection.objects.link(obj)
        obj.location=location
        obj.rotation_euler=(Vector((0,0,.23))-obj.location).to_track_quat("-Z","Y").to_euler()
        return obj,item
    camera,lens=object_at("CAMERA","Comparison camera",(0,-1.95,.47))
    lens.type="ORTHO"
    lens.ortho_scale=1.03
    scene.camera=camera
    ink=bpy.data.materials.new("Title ink")
    ink.use_nodes=True
    nodes=ink.node_tree.nodes
    shade=nodes.new("ShaderNodeEmission")
    shade.inputs["Color"].default_value=(.07,.11,.16,1)
    ink.node_tree.links.new(shade.outputs[0],nodes["Material Output"].inputs[0])
    for text,x,lift,size in [("Lavender  |  Matched wind load",0,.263,.030),
                              ("90 MPa",-.23,-.253,.029),("360 MPa",.23,-.253,.029)]:
        font=bpy.data.curves.new(text,"FONT")
        font.body, font.align_x, font.size=text,"CENTER",size
        glyph=bpy.data.objects.new(text,font)
        scene.collection.objects.link(glyph)
        glyph.parent=camera
        glyph.location=(x,lift,-.6)
        font.materials.append(ink)
    for name,loc,energy,size in [("Key",(-.6,-.70,.95),45,1.0),("Fill",(.65,-.45,.65),24,.8),
                                  ("Rim",(.05,.55,.75),28,.65)]:
        _,light=object_at("LIGHT",name,loc)
        light.energy=energy
        light.shape="DISK"
        light.size=size
    scene.world=bpy.data.worlds.new("Studio daylight")
    scene.world.use_nodes=True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value=(.85,.88,.90,1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value=.55
    scene.render.film_transparent=True
    bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,-.004))
    floor=bpy.context.object
    material=bpy.data.materials.new("Studio matte")
    material.diffuse_color=(.76,.81,.82,1)
    floor.data.materials.append(material)
    scene.render.engine="CYCLES"
    scene.cycles.samples=6
    scene.cycles.use_denoising=True
    scene.render.resolution_x=864
    scene.render.resolution_y=510
    scene.render.resolution_percentage=100
    scene.render.fps=25
    scene.frame_end=len(times)
    scene.frame_step=2
    scene.render.threads_mode="FIXED"
    scene.render.threads=12
    scene.render.image_settings.file_format="PNG"
    scene.render.filepath=str(output/"frame_")
    scene.frame_set(1+int(np.argmax(stems[0,stats["longest_beam_index"]])))
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(output/"lavender_comparison.blend"))
    bpy.ops.render.render(animation=True)


if __name__=="__main__":
    main(*(Path(a).resolve() for a in sys.argv[sys.argv.index("--")+1:]))

