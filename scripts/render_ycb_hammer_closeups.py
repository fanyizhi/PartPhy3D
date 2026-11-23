"""Render COM-follow close-ups of the YCB hammer's solved rigid-body rotations."""
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector
import numpy as np


def main(geometry, tracks, output):
    output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene
    with np.load(geometry) as g:
        vertices,faces,head,centres,head_point=[g[k] for k in ("vertices","faces","head","centres","head_point")]
    with np.load(tracks) as data:
        poses=data["rotation"]
    # View along the thin axis of the scan, using y as downrange and z as up.
    to_display=Matrix(((0,1,0),(-1,0,0),(0,0,1)))
    canvas_centres=[-.44,0,.44]

    def material(name,color,metallic=0.):
        mat=bpy.data.materials.new(name)
        mat.use_nodes=True
        shader=mat.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value=(*color,1)
        shader.inputs["Roughness"].default_value=.29 if metallic else .5
        shader.inputs["Metallic"].default_value=metallic
        return mat

    iron=material("Original steel region",(.18,.23,.27),.75)
    alu=material("Aluminum head edit",(.48,.57,.64),.65)
    shaft=material("Handle region",(.32,.16,.075))
    uniform=material("Uniform density display",(.27,.33,.39),.25)
    accent=[material("COM neutral",(.17,.25,.34)),material("COM assigned",(.02,.43,.40)),
            material("COM edit",(.71,.29,.115))]
    fiducial=material("Head fiducial",(.82,.60,.17),.2)
    for i,shift in enumerate(canvas_centres):
        basis=(vertices-centres[i])
        local=[tuple(to_display@Vector(v)) for v in basis]
        geo=bpy.data.meshes.new(f"YCB scan {i}")
        geo.from_pydata(local,[],faces.tolist())
        geo.update()
        obj=bpy.data.objects.new(f"Hammer {i}",geo)
        scene.collection.objects.link(obj)
        obj.location=(shift,0,.06)
        for m in (uniform if i==0 else shaft,uniform if i==0 else [iron,iron,alu][i]):
            geo.materials.append(m)
        for poly,is_head in zip(geo.polygons,head):
            poly.material_index=int(is_head)
            poly.use_smooth=True
        obj.rotation_mode="QUATERNION"
        for f,k in enumerate(range(0,len(poses[i]),10),1):
            rotation=to_display@Matrix(poses[i,k].tolist())@to_display.transposed()
            obj.rotation_quaternion=rotation.to_quaternion()
            obj.keyframe_insert(data_path="rotation_quaternion",frame=f)
        # Move markers toward the viewing plane solely to avoid hiding them inside the mesh.
        for anchor,m,radius in [(np.zeros(3),accent[i],.010),(head_point-centres[i],fiducial,.007)]:
            marker=to_display@Vector(anchor)
            bpy.ops.mesh.primitive_uv_sphere_add(segments=20,ring_count=12,radius=radius)
            ball=bpy.context.object
            ball.data.materials.append(m)
            for f,k in enumerate(range(0,len(poses[i]),10),1):
                rotation=to_display@Matrix(poses[i,k].tolist())@to_display.transposed()
                v=rotation@marker
                ball.location=(shift+v.x,-.075,.06+v.z)
                ball.keyframe_insert(data_path="location",frame=f)

    bpy.ops.object.camera_add(location=(0,-2.9,.15))
    camera=bpy.context.object
    camera.rotation_euler=(Vector((0,0,.045))-camera.location).to_track_quat("-Z","Y").to_euler()
    camera.data.type="ORTHO"
    camera.data.ortho_scale=1.40
    scene.camera=camera
    for loc,power,size in [((-.55,-.65,1.1),95,1.),((.72,-.5,.65),62,.8),((0,.7,.9),100,.8)]:
        bpy.ops.object.light_add(type="AREA",location=loc)
        light=bpy.context.object
        light.rotation_euler=(-light.location).to_track_quat("-Z","Y").to_euler()
        light.data.energy=power
        light.data.shape="DISK"
        light.data.size=size
    scene.world=bpy.data.worlds.new("Soft studio")
    scene.world.use_nodes=True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value=(.82,.88,.92,1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value=.65
    scene.render.film_transparent=True
    scene.render.engine="CYCLES"
    scene.cycles.samples=8
    scene.cycles.use_denoising=True
    scene.render.resolution_x=1200
    scene.render.resolution_y=486
    scene.render.resolution_percentage=100
    scene.render.threads_mode="FIXED"
    scene.render.threads=12
    scene.render.image_settings.file_format="PNG"
    scene.render.image_settings.color_mode="RGBA"
    scene.render.filepath=str(output/"closeup_")
    scene.frame_end=49
    bpy.ops.wm.save_as_mainfile(filepath=str(output/"ycb_hammer_closeups.blend"))
    bpy.ops.render.render(animation=True)


if __name__=="__main__":
    main(*(Path(p).resolve() for p in sys.argv[sys.argv.index("--")+1:]))
