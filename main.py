import taichi as ti
ti.init(arch = ti.gpu)

size  = 1024

canvas = ti.Vector.field(3,ti.f32,shape = (size,size))


@ti.func
def palette(d):
	return ti.Vector([0.2,0.7,0.9])*(1-d)+ti.Vector([1.0,0.0,1.0])*(d)

@ti.func
def rotate(v ,a):
	return ti.Vector([v.x*ti.cos(a)+v.y*ti.sin(a),v.x*ti.sin(a)*(-1)+v.y*ti.cos(a)])

@ti.func
def map(v ,t):
	for i in range(8):
		tt = t * 0.2
		k = rotate(ti.Vector([v[0],v[2]]),tt)
		v[0] ,v[2] = k[0],k[1]
		k = rotate(ti.Vector([v[0],v[1]]),tt*1.89)
		v[0] ,v[1] = k[0],k[1]
		v[0] ,v[2] = abs(v[0]),abs(v[2])
		v[0]-= 0.5
		v[2]-= 0.5

	return ti.abs(v).dot(ti.Vector([1.0,1.0,1.0]))/3.0

@ti.func
def rm(ro ,rd ,tt):
	t = 0.0
	col = ti.Vector([0.0,0.0,0.0])
	d = 0.0
	for i in range(64):
		p = ro + rd*t
		d = map(p,tt) * 0.5
		if(d < 0.02 or d > 100):
			break
		col  =col + palette(ti.sqrt(p[0]**2+p[1]**2+(p[2]**2))*0.1)/(400.0*d)

		t  = t + d
	return col

@ti.func 
def	render(x,y,t):
	uv = ti.Vector([x,y])/size*2.0-1.0
	ro = ti.Vector([0.0,0.0,-20.0])
	k = rotate(ti.Vector([ro.x,ro.z]),t)
	ro.x ,ro.z = k.x,k.y
	cf = ti.normalized(-1*ro)
	cs = ti.normalized(ti.cross(cf,ti.Vector([0.0,1.0,0.0])))
	cu = ti.normalized(ti.cross(cf,cs))
	uuv = ro+cf*3. + uv.x*cs + uv.y*cu
	rd = ti.normalized(uuv - ro)
	color = rm(ro,rd,t)
	canvas[x,y] = color

@ti.kernel
def run(t:ti.f32):
	for x,y in ti.ndrange(size,size):
		render(x,y,t)


gui = ti.GUI("canvas",res = size)
time = 0
#result_dir = "/home/zhan/work/taichi/taichi/examples/assignment/result"
#video_manager = ti.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)
while time<200:
	time+=1
	run(time/100.0)
	#pixels_img = canvas.to_numpy()
	#video_manager.write_frame(pixels_img)	
	gui.set_image(canvas)
	gui.show()
#video_manager.make_video(gif=True, mp4=True)
