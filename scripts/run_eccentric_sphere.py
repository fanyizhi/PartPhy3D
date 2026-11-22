"""Mesh-plane-guided adaptive sampling of a designed two-material sphere."""
from pathlib import Path
import json
import numpy as np
import mujoco
from scipy.integrate import solve_ivp
from scipy.spatial import ConvexHull, cKDTree
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from PIL import Image
from tactile3d.physics.mpm_validation import run_mpm

out=Path(__file__).resolve().parents[1]/"paper_assets/eccentric_sphere"
out.mkdir(exist_ok=True)
R,r,d=.05,.022,.022
n=642
z=1-2*(np.arange(n)+.5)/n
phi=np.arange(n)*np.pi*(3-np.sqrt(5))
unit=np.stack([np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z],1)
hull=ConvexHull(unit)
def plane_field(p,radius,centre):
    # Maximum signed outward plane offset. Convex mesh containment is exact.
    planes=hull.equations
    return np.concatenate([((chunk-centre)@planes[:,:3].T+radius*planes[:,3]).max(1)
                           for chunk in np.array_split(p,max(1,len(p)//500))])
def sample():
    h=.008
    axis=np.arange(-7,8)*h
    p=np.stack(np.meshgrid(axis,axis,axis,indexing="ij"),-1).reshape(-1,3)
    ext=plane_field(p,R,0)
    inner=plane_field(p,r,[d,0,0])
    bound=np.sqrt(3)*h/2
    nearby=(ext<bound)&((abs(ext)<=bound)|(abs(inner)<=bound))
    offsets=np.array(np.meshgrid(*[[-h/4,h/4]]*3)).reshape(3,-1).T
    fine=(p[nearby,None]+offsets).reshape(-1,3)
    bulk=p[(ext<0)&~nearby]
    x=np.vstack([bulk,fine])
    vol=np.r_[np.full(len(bulk),h**3),np.full(len(fine),(h/2)**3)]
    take=plane_field(x,R,0)<=0
    x,vol=x[take],vol[take]
    return x,vol,plane_field(x,r,[d,0,0])<=0
x,vol,core=sample()
m=np.where(core,3200.,800.)*vol
com=np.average(x,weights=m,axis=0)
inertia=np.eye(3)*np.sum(m*np.sum((x-com)**2,axis=1))-((x-com)*m[:,None]).T@(x-com)
V=lambda a:4*np.pi*a**3/3
mb,extra=800*V(R),2400*V(r)
M=mb+extra
cx=extra*d/M
Iy=.4*(mb*R**2+extra*r**2)+extra*d*d
t=np.linspace(0,.65,651)
ref=solve_ivp(lambda _,y:[y[1],9.81*M*cx*np.cos(y[0])/Iy],(0,t[-1]),[0,0],
              t_eval=t,rtol=1e-11,atol=1e-13).y[0]
def pivot(centre,tensor):
    vec=lambda a:" ".join(map(str,a))
    full=[tensor[i,j] for i,j in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]]
    xml=f"""<mujoco><option timestep=".001" integrator="RK4"/><worldbody><body>
    <joint type="hinge" axis="0 1 0"/><inertial pos="{vec(centre)}" mass="{m.sum()}"
    fullinertia="{vec(full)}"/><geom type="sphere" size=".05"/>
    </body></worldbody></mujoco>"""
    model=mujoco.MjModel.from_xml_string(xml)
    state=mujoco.MjData(model)
    q=[0.]
    for _ in t[1:]:
        mujoco.mj_step(model,state)
        q.append(state.qpos[0])
    return np.array(q)
seg=pivot(com,inertia)
uniform=pivot(np.zeros(3),np.eye(3)*.4*m.sum()*R**2)
initial=x+[0,0,.060]
data=dict(position_m=initial,mass_kg=m,volume_m3=vol,youngs_pa=np.where(core,120000.,15000.),
          poisson=np.full(len(x),.30))
frames=[initial]
def capture(step,time,pos):
    if step%20==0: frames.append(pos)
res=run_mpm(data,dx=.008,steps=800,timestep_s=.00012,initial_vertical_speed=-1.,
            floor_z_m=0.,on_step=capture)
fig,axes=plt.subplots(1,4,figsize=(12,3.35),gridspec_kw={"width_ratios":[1,1,1,1.35]},layout="constrained")
colors=["#579c9f","#c26739"]
for ax,title in zip(axes,["a   Boundary meshes","b   Adaptive particles","c   MPM impact","d   MuJoCo pivot"]):
    ax.set_title(title,fontsize=11,loc="left")
for ax in axes[:3]:
    ax.set(xlim=(-.064,.064),ylim=(-.062,.07),aspect="equal");ax.axis("off")
for rad,offset,color in [(R,0,"#9daeb2"),(r,d,colors[1])]:
    front=hull.simplices[hull.equations[:,1]<-.01]
    tri=(unit[front]*rad+[offset,0,0])[:,:,[0,2]]
    axes[0].add_collection(LineCollection(tri[:,[0,1,2,0]],colors=color,lw=.34))
cut=abs(x[:,1])<np.cbrt(vol)*.51
axes[1].scatter(x[cut,0],x[cut,2],s=np.cbrt(vol[cut])**2*4e5,
                c=np.where(core[cut],colors[1],colors[0]),lw=0)
axes[0].set_xlabel("Given outer and core boundaries")
axes[1].text(.5,-.02,"4 mm interface  /  8 mm bulk",transform=axes[1].transAxes,ha="center",fontsize=8)
ang=np.linspace(0,2*np.pi,220)
loops=[]
for rad,offset,color in [(R,0,colors[0]),(r,d,colors[1])]:
    base=np.stack([offset+rad*np.cos(ang),np.zeros_like(ang),rad*np.sin(ang)],1)
    dist,idx=cKDTree(x).query(base,k=8)
    w=1/np.maximum(dist,1e-6)**2
    patch=axes[2].add_patch(Polygon(np.zeros((3,2)),fc=color,ec="#425758",lw=.8))
    loops.append((base,idx,w/w.sum(1)[:,None],patch))
axes[2].axhline(-.060,color=".45",lw=1)
clock=axes[2].text(.5,-.02,"",transform=axes[2].transAxes,ha="center",fontsize=8)
ax=axes[3]
for q,c,style,label in [(ref,"#adb6b8","-","Analytic"),(seg,"#176d70","--","Segmented"),(uniform,"#ad633f","-","Uniform, matched mass")]:
    ax.plot(t,np.rad2deg(q),style,c=c,label=label,lw=2)
ax.set(xlabel="Time (s)",ylabel="Rotation (deg)")
ax.legend(frameon=False,fontsize=7)
ax.spines[["top","right"]].set_visible(False)
imgs=[]
for j,pos in enumerate(frames):
    for base,idx,w,patch in loops:
        pts=base+np.sum(w[:,:,None]*(pos-initial)[idx],axis=1)
        patch.set_xy(pts[:,[0,2]])
    clock.set_text(f"{2.4*j:.1f} ms   ·   motion ×1")
    fig.canvas.draw()
    imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
    if j==12:
        for ext in ("png","pdf","svg"): fig.savefig(out/f"sphere.{ext}",dpi=300)
imgs[0].save(out/"sphere.gif",save_all=True,append_images=imgs[1:],duration=75,loop=0)
stats=dict(exact_mass_kg=M,particle_mass_kg=m.sum(),true_com_mm=cx*1000,segmented_com_mm=(com*1000).tolist(),
           segmented_angle_rmse_deg=np.rad2deg(np.mean((seg-ref)**2)**.5),
           uniform_angle_rmse_deg=np.rad2deg(np.mean((uniform-ref)**2)**.5),
           particles=len(x),fine_particles=int(np.sum(vol<.006**3)),core_particles=int(core.sum()),
           final_min_J=res["jacobian"].min(),mass_transfer_residual_kg=res["max_p2g_mass_residual_kg"])
(out/"results.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
np.savez_compressed(out/"simulation.npz",position_m=x,volume_m3=vol,mass_kg=m,core=core,
                    youngs_pa=data["youngs_pa"],poisson=data["poisson"],impact_positions_m=frames,
                    pivot_time_s=t,segmented_angle_rad=seg,uniform_angle_rad=uniform,reference_angle_rad=ref)
print(stats)
