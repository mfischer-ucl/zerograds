import os
import drjit as dr
import mitsuba as mi
from mitsuba.scalar_rgb import Transform4f as T


def setup_dice_scene(hparams, simplescene=False):
    scenefolder = hparams['mts_scenefolder']

    supersimple = False
    use_colored = False

    if simplescene:
        xmlpath = f'{scenefolder}/stacked_dice_simple/stacked_dice_simple_autoReadable.xml'
    elif supersimple:
        xmlpath = f'{scenefolder}/stacked_dice_simple/stacked_dice_supersimple_autoReadable.xml'
    else:
        if use_colored is False:
            xmlpath = f'{scenefolder}/stacked_dice/stacked_dice_autoReadable.xml'
        else:
            xmlpath = f'{scenefolder}/stacked_dice/stacked_dice_colored_autoReadable.xml'
    scene = create_scene_from_xml(xmlpath, resx=hparams['resx'], resy=hparams['resy'], integrator=hparams['integrator'],
                                  maxdepth=hparams['max_depth'], reparam_max_depth=hparams['reparam_max_depth'])
    params = mi.traverse(scene)

    if simplescene:
        mat_id = ['PLYMesh.vertex_positions', 'PLYMesh_1.vertex_positions']
    elif supersimple:
        mat_id = ['PLYMesh.vertex_positions']
    else:
        if use_colored:
            ids = [''] + [f'_{x}' for x in range(1, 5)]
            mat_id = [f'PLYMesh{x}.vertex_positions' for x in ids]
        else:
            ids = [''] + [f'_{x}' for x in range(1, 10)]
            mat_id = [f'PLYMesh{x}.vertex_positions' for x in ids]
    initial_vertex_positions = [dr.unravel(mi.Point3f, params[x]) for x in mat_id]
    return scene, params, mat_id, initial_vertex_positions


def setup_mesh_scene(hparams):
    xmlpath = f"{hparams['mts_scenefolder']}/meshtest/meshtest_autoReadable.xml"
    scene = create_scene_from_xml(xmlpath, resx=hparams['resx'], resy=hparams['resy'], integrator=hparams['integrator'],
                                  maxdepth=hparams['max_depth'], reparam_max_depth=hparams['reparam_max_depth'])
    params = mi.traverse(scene)
    print(params)

    mat_id = 'PLYMesh.vertex_positions'
    initial_vertex_positions = dr.unravel(mi.Point3f, params[mat_id])
    return scene, params, mat_id, initial_vertex_positions


def setup_sphere_scene(hparams, simplescene=False):
    if simplescene is False:
        scene = create_sphere_scene(hparams)
    else:
        scene = create_sphere_scene_simple(hparams)
    params = mi.traverse(scene)
    mat_id = 'sphere.vertex_positions'
    initial_vertex_positions = dr.unravel(mi.Point3f, params[mat_id])
    # initial_vertex_positions = dr.unravel(mi.TensorXf, params[mat_id])
    return scene, params, mat_id, initial_vertex_positions


def create_sphere_scene(hparams):

    scenefolder = hparams['mts_scenefolder']
    print(f'{scenefolder}/sphere_scene/meshes/rectangle.obj',)
    if hparams['integrator'] == 'path':
        integrator = {'type': 'path', 'max_depth': hparams['max_depth']}
    elif hparams['integrator'] == 'prb_reparam':
        integrator = {'type': 'prb_reparam',
                      'max_depth': hparams['max_depth'],
                      'reparam_max_depth': hparams['reparam_max_depth']}
    elif hparams['integrator'] == 'direct':
        integrator = {'type': 'direct'}
    else: raise ValueError("Unknown integrator {} - choose from ['path', 'prb_reparam']".format(hparams['integrator']))
    spherepos = [0.0, 0.0, -3.2]
    lightpos = [-4.5, 4.5, 2.0]
    lightInt = 1e3

    scene = mi.load_dict({
        'type': 'scene',
        'integrator': integrator,
        'sensor': {
            'type': 'perspective',
            'to_world': T.look_at(
                origin=(0, 0, 2),
                target=(0, 0, 0),
                up=(0, 1, 0)
            ),
            'fov': 60,
            'film': {
                'type': 'hdrfilm',
                'width': hparams['res_x'],
                'height': hparams['res_y'],
                'rfilter': {'type': 'gaussian'},
                'sample_border': True
            },
        },
        'wall': {
            'type': 'obj',
            'filename': f'{scenefolder}/sphere_scene/meshes/rectangle.obj',
            'to_world': T.translate([0, 0, -4]).scale(2.0),
            'face_normals': True,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': (0.8, 0.8, 0.8)},
            }
        },
        'wall_bottom': {
            'type': 'obj',
            'filename': f'{scenefolder}/sphere_scene/meshes/rectangle.obj',
            'to_world': T.translate([0.0, -2.0, -2.0]).rotate([1, 0, 0], -90).scale(2.0),
            'face_normals': True,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': (0.4, 0.4, 0.4)},
            }
        },
        'wall_right': {
            'type': 'obj',
            'filename': f'{scenefolder}/sphere_scene/meshes/rectangle.obj',
            'to_world': T.translate([2.0, 0.0, -2.0]).rotate([0, 1, 0], -90).scale(2.0),
            'face_normals': True,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': (0.0, 1.0, 0.0)},
            }
        },
        'light': {
            'type': 'obj',
            'filename': f'{scenefolder}/sphere_scene/meshes/sphere.obj',
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'rgb', 'value': [lightInt, lightInt, lightInt]}
            },
            'to_world': T.translate(lightpos).scale(0.25)
        },
        'sphere': {
            'type': 'obj',
            'filename': f'{scenefolder}/sphere_scene/meshes/sphere.obj',
            'to_world': T.translate(spherepos).scale(0.5),
            'face_normals': True,
            # 'bsdf': {
            #     'type': 'dielectric',
            #     'id': 'simple-glass-bsdf',
            #     'ext_ior': 'air',
            #     'int_ior': 1.5,
            #     'specular_reflectance': {'type': 'spectrum', 'value': 0},
            # }
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': (1.0, 0.0, 0.0)},
            }
        }
    })

    return scene


def create_sphere_scene_simple(hparams):

    scenefolder = hparams['mts_scenefolder']
    print(f'{scenefolder}/sphere_scene/meshes/rectangle.obj',)
    if hparams['integrator'] == 'path':
        integrator = {'type': 'path', 'max_depth': hparams['max_depth']}
    elif hparams['integrator'] == 'prb_reparam':
        integrator = {'type': 'prb_reparam',
                      'max_depth': hparams['max_depth'],
                      'reparam_max_depth': hparams['reparam_max_depth']}
    elif hparams['integrator'] == 'direct':
        integrator = {'type': 'direct'}
    else: raise ValueError("Unknown integrator {} - choose from ['path', 'prb_reparam']".format(hparams['integrator']))
    spherepos = [0.0, 0.0, -3.2]
    lightpos = [-4.5, 4.5, 2.0]
    lightInt = 1e3

    scene = mi.load_dict({
        'type': 'scene',
        'integrator': integrator,
        'sensor': {
            'type': 'perspective',
            'to_world': T.look_at(
                origin=(0, 0, 2),
                target=(0, 0, 0),
                up=(0, 1, 0)
            ),
            'fov': 60,
            'film': {
                'type': 'hdrfilm',
                'width': hparams['res_x'],
                'height': hparams['res_y'],
                'rfilter': {'type': 'gaussian'},
                'sample_border': True
            },
        },
        'wall': {
            'type': 'obj',
            'filename': f'{scenefolder}/sphere_scene/meshes/rectangle.obj',
            'to_world': T.translate([0, 0, -4]).scale(5.0),
            'face_normals': True,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': (0.8, 0.8, 0.8)},
            }
        },
        'light': {
            'type': 'obj',
            'filename': f'{scenefolder}/sphere_scene/meshes/sphere.obj',
            'emitter': {
                'type': 'area',
                'radiance': {'type': 'rgb', 'value': [lightInt, lightInt, lightInt]}
            },
            'to_world': T.translate(lightpos).scale(0.25)
        },
        'sphere': {
            'type': 'obj',
            'filename': f'{scenefolder}/sphere_scene/meshes/sphere.obj',
            'to_world': T.translate(spherepos).scale(0.5),
            'face_normals': True,
            # 'bsdf': {
            #     'type': 'dielectric',
            #     'id': 'simple-glass-bsdf',
            #     'ext_ior': 'air',
            #     'int_ior': 1.5,
            #     'specular_reflectance': {'type': 'spectrum', 'value': 0},
            # }
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': (1.0, 0.0, 0.0)},
            }
        }
    })

    return scene


def setup_caustic_scene(hparams):
    scenefolder = hparams['mts_scenefolder']

    CONFIGS = {
        'circle': {
            'emitter': 'gray',
            'reference': os.path.join(scenefolder, 'references/circle-512.jpg'),
        },
        'wave': {
            'emitter': 'gray',
            'reference': os.path.join(scenefolder, 'references/wave-1024.jpg'),
        },
        'sunday': {
            'emitter': 'bayer',
            'reference': os.path.join(scenefolder, 'references/sunday-512.jpg'),
        },
        'siggraph': {
            'emitter': 'bayer',
            'reference': os.path.join(scenefolder, 'references/siggraph-512.jpg'),
        },
        'starrynight': {
            'emitter': 'bayer',
            'reference': os.path.join(scenefolder, 'references/starrynight-512.jpg'),
        },
    }

    config = CONFIGS[hparams['config']]
    res = hparams['optim_dim']
    config.update({
        'render_resolution': (res, res),
        'heightmap_resolution': (res, res),
        'n_upsampling_steps': 0,
        'spp': 32,
        'max_iterations': 1000,
        'learning_rate': 3e-5,
    })

    emitter = None
    if config['emitter'] == 'gray':
        emitter = {
            'type': 'directionalarea',
            'radiance': {
                'type': 'spectrum',
                'value': 0.8
            },
        }
    elif config['emitter'] == 'bayer':
        bayer = dr.zeros(mi.TensorXf, (32, 32, 3))
        bayer[::2, ::2, 2] = 2.2
        bayer[::2, 1::2, 1] = 2.2
        bayer[1::2, 1::2, 0] = 2.2

        emitter = {
            'type': 'directionalarea',
            'radiance': {
                'type': 'bitmap',
                'bitmap': mi.Bitmap(bayer),
                'raw': True,
                'filter_type': 'nearest'
            },
        }

    integrator = {
        'type': 'ptracer',
        'samples_per_pass': 256,
        'max_depth': 4,
        'hide_emitters': False,
    }

    # Looking at the receiving plane, not looking through the lens
    sensor_to_world = mi.ScalarTransform4f.look_at(
        target=[0, -20, 0],
        origin=[0, -4.65, 0],
        up=[0, 0, 1]
    )
    resx, resy = config['render_resolution']
    sensor = {
        'type': 'perspective',
        'near_clip': 1,
        'far_clip': 1000,
        'fov': 45,
        'to_world': sensor_to_world,

        'sampler': {
            'type': 'independent',
            'sample_count': 512  # Not really used
        },
        'film': {
            'type': 'hdrfilm',
            'width': resx,
            'height': resy,
            'pixel_format': 'rgb',
            'rfilter': {
                # Important: smooth reconstruction filter with a footprint larger than 1 pixel.
                'type': 'gaussian'
            }
        },
    }

    lens_res = config.get('lens_res', config['heightmap_resolution'])
    lens_fname = os.path.join(scenefolder, 'lens_{}_{}.ply'.format(*lens_res))
    if not os.path.isfile(lens_fname):
        m = create_flat_lens_mesh(lens_res)
        m.write_ply(lens_fname)
        print('[+] Wrote lens mesh ({}x{} tesselation) file to: {}'.format(*lens_res, lens_fname))

    scene = {
        'type': 'scene',
        'sensor': sensor,
        'integrator': integrator,
        # Glass BSDF
        'simple-glass': {
            'type': 'dielectric',
            'id': 'simple-glass-bsdf',
            'ext_ior': 'air',
            'int_ior': 1.5,
            'specular_reflectance': {'type': 'spectrum', 'value': 0},
        },
        'white-bsdf': {
            'type': 'diffuse',
            'id': 'white-bsdf',
            'reflectance': {'type': 'rgb', 'value': (1, 1, 1)},
        },
        'black-bsdf': {
            'type': 'diffuse',
            'id': 'black-bsdf',
            'reflectance': {'type': 'spectrum', 'value': 0},
        },
        # Receiving plane
        'receiving-plane': {
            'type': 'obj',
            'id': 'receiving-plane',
            'filename': os.path.join(scenefolder, 'meshes/rectangle.obj'),
            'to_world': \
                mi.ScalarTransform4f.look_at(
                    target=[0, 1, 0],
                    origin=[0, -7, 0],
                    up=[0, 0, 1]
                ).scale((5, 5, 5)),
            'bsdf': {'type': 'ref', 'id': 'white-bsdf'},
        },
        # Glass slab, excluding the 'exit' face (added separately below)
        'slab': {
            'type': 'obj',
            'id': 'slab',
            'filename': os.path.join(scenefolder, 'meshes/slab.obj'),
            'to_world': mi.ScalarTransform4f.rotate(axis=(1, 0, 0), angle=90),
            'bsdf': {'type': 'ref', 'id': 'simple-glass'},
        },
        # Glass rectangle, to be optimized
        'lens': {
            'type': 'ply',
            'id': 'lens',
            'filename': lens_fname,
            'to_world': mi.ScalarTransform4f.rotate(axis=(1, 0, 0), angle=90),
            'bsdf': {'type': 'ref', 'id': 'simple-glass'},
        },

        # Directional area emitter placed behind the glass slab
        'focused-emitter-shape': {
            'type': 'obj',
            'filename': os.path.join(scenefolder, 'meshes/rectangle.obj'),
            'to_world': mi.ScalarTransform4f.look_at(
                target=[0, 0, 0],
                origin=[0, 5, 0],
                up=[0, 0, 1]
            ),
            'bsdf': {'type': 'ref', 'id': 'black-bsdf'},
            'focused-emitter': emitter,
        },
    }

    scene = mi.load_dict(scene)

    # load ref img
    # sensor = scene.sensors()[0]
    # crop_size = sensor.film().crop_size()
    # image_ref = load_ref_image(config, crop_size, output_dir=output_dir)

    initial_heightmap_resolution = config['heightmap_resolution']
    heightmap_texture = mi.load_dict({
        'type': 'bitmap',
        'id': 'heightmap_texture',
        'bitmap': mi.Bitmap(dr.zeros(mi.TensorXf, initial_heightmap_resolution)),
        'raw': True,
    })

    params = mi.traverse(heightmap_texture)
    params_scene = mi.traverse(scene)
    mat_id = 'lens.vertex_positions'

    return scene, [params, params_scene], mat_id


def setup_brdf_scene(hparams):
    scenefolder = hparams['mts_scenefolder']
    xmlpath = f'{scenefolder}/brdf/scene_autoreadable.xml'
    scene = create_scene_from_xml(xmlpath, resx=hparams['resx'], resy=hparams['resy'], integrator=hparams['integrator'],
                                  maxdepth=hparams['max_depth'], reparam_max_depth=hparams['reparam_max_depth'])
    params = mi.traverse(scene)
    mat_id = ['bsdf-matpreview.diffuse_reflectance.value', 'bsdf-matpreview.eta']
    return scene, params, mat_id


def setup_cbox_scene(hparams, inset_only):
    scenefolder = hparams['mts_scenefolder']
    if inset_only:
        xmlpath = f'{scenefolder}/diff_rendering/cbox_aspect_cameraCloser_autoReadable.xml'
    else:

        if 'highdim' in hparams['experiment_mode']:
            xmlpath = f'{scenefolder}/diff_rendering/cbox_aspect_highdim_autoReadable.xml'
        else:
            xmlpath = f'{scenefolder}/diff_rendering/cbox_aspect_autoReadable.xml'
    scene = create_scene_from_xml(xmlpath, resx=hparams['resx'], resy=hparams['resy'], integrator=hparams['integrator'],
                                  maxdepth=hparams['max_depth'], reparam_max_depth=hparams['reparam_max_depth'])
    params = mi.traverse(scene)

    if 'highdim' in hparams['experiment_mode']:
        mat_id = ['PLYMesh_{}.vertex_positions'.format(x) for x in range(6, 76)]
        initial_vertex_positions = [dr.unravel(mi.Point3f, params[m]) for m in mat_id]
        print("NUMBER OF IDs for HIGHDIM CUBE:", len(mat_id))
        print(params)
    else:
        mat_id = ['PLYMesh.vertex_positions', 'mat-LeftWallBSDF.brdf_0.reflectance.value', 'PLYMesh_7.vertex_positions',
                  'PLYMesh_6.vertex_positions']
        initial_vertex_positions = [dr.unravel(mi.Point3f, params[mat_id[0]]),
                                    dr.unravel(mi.Point3f, params[mat_id[2]]),
                                    dr.unravel(mi.Point3f, params[mat_id[3]])]

    return scene, params, mat_id, initial_vertex_positions


def setup_planck_scene(hparams, meshpath=None):
    scenefolder = hparams['mts_scenefolder']
    # xmlpath = f'{scenefolder}/mesh_planck/planck_autoReadable.xml'
    xmlpath = f'{scenefolder}/mesh_planck/planck_autoReadable_fewer.xml'
    scene = create_scene_from_xml(xmlpath, resx=hparams['resx'], resy=hparams['resy'], integrator=hparams['integrator'],
                                  maxdepth=hparams['max_depth'], reparam_max_depth=hparams['reparam_max_depth'])
    params = mi.traverse(scene)
    mat_id = 'PLYMesh.vertex_positions'
    initial_vertex_positions = dr.unravel(mi.Point3f, params[mat_id])
    return scene, params, mat_id, initial_vertex_positions


def create_scene_from_xml(xmlpath,
                          resx=512, resy=512,
                          integrator='path', maxdepth=6, reparam_max_depth=2):
    lines = open(xmlpath, 'r').readlines()
    for idx in range(len(lines)):
        line = lines[idx]
        if 'resx' in lines[idx]:
            lines[idx] = line.replace('resolution_x', str(resx))
        if 'resy' in lines[idx]:
            lines[idx] = line.replace('resolution_y', str(resy))
        if 'integrator' in lines[idx]:
            lines[idx] = line.replace('integrator_type', integrator)
        if 'max_depth' in lines[idx]:
            lines[idx] = line.replace('depth_value', str(maxdepth))
        if 'reparam_max_depth' in lines[idx]:
            if integrator == 'prb_reparam':
                lines[idx] = line.replace('reparam_depth_value', str(reparam_max_depth))
            else:
                lines[idx] = ''

    tmppath = os.path.join(os.path.split(xmlpath)[0], 'tmp.xml')
    open(tmppath, 'w').writelines(lines)
    scene = mi.load_file(tmppath)
    # os.remove(tmppath)
    return scene



# copied from caustics example:
def create_flat_lens_mesh(resolution):
    # Generate UV coordinates
    U, V = dr.meshgrid(
        dr.linspace(mi.Float, 0, 1, resolution[0]),
        dr.linspace(mi.Float, 0, 1, resolution[1]),
        indexing='ij'
    )
    texcoords = mi.Vector2f(U, V)

    # Generate vertex coordinates
    X = 2.0 * (U - 0.5)
    Y = 2.0 * (V - 0.5)
    vertices = mi.Vector3f(X, Y, 0.0)

    # Create two triangles per grid cell
    faces_x, faces_y, faces_z = [], [], []
    for i in range(resolution[0] - 1):
        for j in range(resolution[1] - 1):
            v00 = i * resolution[1] + j
            v01 = v00 + 1
            v10 = (i + 1) * resolution[1] + j
            v11 = v10 + 1
            faces_x.extend([v00, v01])
            faces_y.extend([v10, v10])
            faces_z.extend([v01, v11])

    # Assemble face buffer
    faces = mi.Vector3u(faces_x, faces_y, faces_z)

    # Instantiate the mesh object
    mesh = mi.Mesh("lens-mesh", resolution[0] * resolution[1], len(faces_x), has_vertex_texcoords=True)

    # Set its buffers
    mesh_params = mi.traverse(mesh)
    mesh_params['vertex_positions'] = dr.ravel(vertices)
    mesh_params['vertex_texcoords'] = dr.ravel(texcoords)
    mesh_params['faces'] = dr.ravel(faces)
    mesh_params.update()

    return mesh


def load_ref_image(config, resolution, output_dir):
    b = mi.Bitmap(config['reference'])
    b = b.convert(mi.Bitmap.PixelFormat.RGB, mi.Bitmap.Float32, False)
    if b.size() != resolution:
        b = b.resample(resolution)

    mi.util.write_bitmap(join(output_dir, 'out_ref.exr'), b)

    print('[i] Loaded reference image from:', config['reference'])
    return mi.TensorXf(b)
