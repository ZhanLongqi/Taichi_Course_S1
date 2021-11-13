import taichi as ti
import numpy as np
import argparse
from ray_tracing_models import Ray, Camera, Hittable_list, Sphere\
    , PI, random_in_unit_sphere, refract, reflect, reflectance,\
         random_unit_vector 
ti.init(arch=ti.gpu)

# Canvas
aspect_ratio = 1.0
image_width = 800
image_height = int(image_width / aspect_ratio)
canvas = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))
color_swap = ti.Vector.field(3,dtype=ti.f32,shape=(image_width,image_height))
pos_map = ti.Vector.field(3,dtype=ti.f32,shape=(image_width,image_height))
pos_swap = ti.Vector.field(3,dtype=ti.f32,shape=(image_width,image_height))
normal_map = ti.Vector.field(3,dtype=ti.f32,shape=(image_width,image_height))
normal_swap = ti.Vector.field(3,dtype=ti.f32,shape=(image_width,image_height))
id_map = ti.Vector.field(1,dtype=ti.i32,shape=(image_width,image_height))
id_swap = ti.Vector.field(1,dtype=ti.i32,shape=(image_width,image_height))
gui = ti.GUI("Ray Tracing", res=(image_width, image_height))
camera = Camera()
# Rendering parameters
samples_per_pixel = 1
blend_ratio = 0.8
max_depth = 1
sample_on_unit_sphere_surface = True
last_cursor_pos = ti.Vector.field(2,dtype=ti.f32,shape=())

@ti.kernel
def render():
    for i, j in canvas:
        u = (i + ti.random()) / image_width
        v = (j + ti.random()) / image_height
        color = ti.Vector([0.0, 0.0, 0.0])
        initial = True
        for n in range(samples_per_pixel):
            ray = camera.get_ray(u, v)
            color+= ray_color(ray,i ,j ,initial)
            initial = False
        color /= samples_per_pixel
        canvas[i, j] =  canvas[i,j]*blend_ratio + color*(1-blend_ratio)
        #canvas[i,j]= ti.sqrt(color[0]**2+color[1]**2+color[2]**2)*ti.Vector([1.0,1.0,1.0])

@ti.kernel
def swap():
    for i,j in canvas:
        color_swap[i,j] = canvas[i,j]
        pos_swap[i,j] = pos_map[i,j]
        normal_swap[i,j] = normal_map[i,j]
        id_swap[i,j] = id_map[i,j]

# Path tracing
@ti.func
def ray_color(ray ,i ,j, initial):
    color_buffer = ti.Vector([0.0, 0.0, 0.0])
    brightness = ti.Vector([1.0, 1.0, 1.0])
    scattered_origin = ray.origin
    scattered_direction = ray.direction
    p_RR = 0.8
    for n in range(max_depth):
        if ti.random() > p_RR:
            break
        is_hit, hit_point, hit_point_normal, front_face, material, color ,obj_id \
            = scene.hit(Ray(scattered_origin, scattered_direction))
        if(initial):
            pos_map[i,j] = hit_point
            normal_map[i,j] = hit_point_normal
            id_map[i,j][0] = obj_id

        if is_hit:
            #color_buffer = float(obj_id)/(Sphere.id_pool) * ti.Vector([1.0,1.0,1.0])
            #color_buffer = hit_point + ti.Vector([1.0,1.0,1.0])
            #color_buffer = color_buffer / 2
            if material == 0:
                color_buffer = color * brightness
                break
            else:
                # Diffuse
                if material == 1:
                    target = hit_point + hit_point_normal
                    if sample_on_unit_sphere_surface:
                        target += random_unit_vector()
                    else:
                        target += random_in_unit_sphere()
                    scattered_direction = target - hit_point
                    scattered_origin = hit_point
                    brightness *= color
                # Metal and Fuzz Metal
                elif material == 2 or material == 4:
                    fuzz = 0.0
                    if material == 4:
                        fuzz = 0.4
                    scattered_direction = reflect(scattered_direction.normalized(),
                                                  hit_point_normal)
                    if sample_on_unit_sphere_surface:
                        scattered_direction += fuzz * random_unit_vector()
                    else:
                        scattered_direction += fuzz * random_in_unit_sphere()
                    scattered_origin = hit_point
                    if scattered_direction.dot(hit_point_normal) < 0:
                        break
                    else:
                        brightness *= color
                # Dielectric
                elif material == 3:
                    refraction_ratio = 1.5
                    if front_face:
                        refraction_ratio = 1 / refraction_ratio
                    cos_theta = min(-scattered_direction.normalized().dot(hit_point_normal), 1.0)
                    sin_theta = ti.sqrt(1 - cos_theta * cos_theta)
                    # total internal reflection
                    if refraction_ratio * sin_theta > 1.0 or reflectance(cos_theta, refraction_ratio) > ti.random():
                        scattered_direction = reflect(scattered_direction.normalized(), hit_point_normal)
                    else:
                        scattered_direction = refract(scattered_direction.normalized(), hit_point_normal, refraction_ratio)
                    scattered_origin = hit_point
                    brightness *= color
                brightness /= p_RR
    return color_buffer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Naive Ray Tracing')
    parser.add_argument(
        '--max_depth', type=int, default=10, help='max depth (default: 10)')
    parser.add_argument(
        '--samples_per_pixel', type=int, default=4, help='samples_per_pixel  (default: 4)')
    parser.add_argument(
        '--samples_in_unit_sphere', action='store_true', help='whether sample in a unit sphere')
    args = parser.parse_args()

    max_depth = args.max_depth
    samples_per_pixel = args.samples_per_pixel
    sample_on_unit_sphere_surface = not args.samples_in_unit_sphere
    scene = Hittable_list()

    # Light source
    scene.add(Sphere(center=ti.Vector([0, 5.4, -1]), radius=3.0, material=0, color=ti.Vector([10.0, 10.0, 10.0])))
    # Ground
    scene.add(Sphere(center=ti.Vector([0, -100.5, -1]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
    # ceiling
    scene.add(Sphere(center=ti.Vector([0, 102.5, -1]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
    # back wall
    scene.add(Sphere(center=ti.Vector([0, 1, 101]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
    # right wall
    scene.add(Sphere(center=ti.Vector([-101.5, 0, -1]), radius=100.0, material=1, color=ti.Vector([0.6, 0.0, 0.0])))
    # left wall
    scene.add(Sphere(center=ti.Vector([101.5, 0, -1]), radius=100.0, material=1, color=ti.Vector([0.0, 0.6, 0.0])))

    # Diffuse ball
    scene.add(Sphere(center=ti.Vector([0, -0.2, -1.5]), radius=0.3, material=1, color=ti.Vector([0.8, 0.3, 0.3])))
    # Metal ball
    scene.add(Sphere(center=ti.Vector([-0.8, 0.2, -1]), radius=0.7, material=2, color=ti.Vector([0.6, 0.8, 0.8])))
    # Glass ball
    scene.add(Sphere(center=ti.Vector([0.7, 0, -0.5]), radius=0.5, material=3, color=ti.Vector([1.0, 1.0, 1.0])))
    # Metal ball-2
    scene.add(Sphere(center=ti.Vector([0.6, -0.3, -2.0]), radius=0.2, material=4, color=ti.Vector([0.8, 0.6, 0.2])))



    canvas.fill(0)
    cnt = 0
    while gui.running:
        print("pos",gui.get_cursor_pos(),end="\r")
        pressed = gui.is_pressed(ti.GUI.LMB)
        if gui.get_event(ti.GUI.PRESS) or pressed:
            if pressed:
                diffx,diffy = gui.get_cursor_pos()[0]-last_cursor_pos[None][0],gui.get_cursor_pos()[1]-last_cursor_pos[None][1]
                camera.lookat[None][0]-= diffx * camera.step
                camera.lookat[None][1]+= diffy * camera.step
            if gui.event.key == 'w':
                camera.lookfrom[None][0]+=camera.step
        last_cursor_pos[None][0] = gui.get_cursor_pos()[0]
        last_cursor_pos[None][1] = gui.get_cursor_pos()[1]
        camera.react()
        #print(gui.get_cursor_pos(),end="\r")
        render()
        #swap()
        cnt = 1
        gui.set_image(np.sqrt(canvas.to_numpy() / cnt))
        gui.show()
