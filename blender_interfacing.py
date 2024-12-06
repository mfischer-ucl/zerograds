import os
import time
import torch
import socket
import numpy as np
import torchvision.io as io
from subprocess import Popen
import matplotlib.pyplot as plt


def bool2int(b):
    if isinstance(b, bool):
        return 1 if b else 0
    else: return b


def init_blender(path_to_blender, blendfilepath, port_id, shmpath, suppress_output=True, sleepsecs=2.0):

    # start non-blocking subprocess with blender in the background
    # this will execute the render_server script in blender, which listens to the port and renders.
    process = Popen(['{} -b {}'
                     ' --python-text renderAutomatic -- {} {} {}'.format(path_to_blender,
                                                                         blendfilepath,
                                                                         port_id,
                                                                         shmpath,
                                                                         ' >/dev/null 2>&1' if suppress_output else '')]
                    , shell=True)

    # give blender some time to load and run the script that opens the server port.
    # without this, we will get OSError[111]: Connection Refused.
    time.sleep(sleepsecs)
    print("Openend blender...")

    host = socket.gethostname()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port_id))
    return s


def get_tensor(path, int_id, is_batched, batchsize):
    if is_batched is False:
        img = io.read_image(os.path.join(path, 'img{}.png'.format(int_id)), mode=io.ImageReadMode.RGB)
        return img.float().unsqueeze(0) / 255.
    else:
        imgs = [io.read_image(os.path.join(path, f'img{int_id}_{j}.png'),
                              mode=io.ImageReadMode.RGB) for j in range(batchsize)]
        return torch.stack(imgs).float() / 255.


def render_blender(values, active_socket, argstring_factory, shmpath,
                   report_timing=False, verbose=False, device='cuda', is_batched=False, batchsize=13):
    # shmpath is the path relative to /dev/shm, the ramdisk we're using

    start = time.time()

    # make string from dict values
    argstring = argstring_factory(values)

    if isinstance(argstring, str):
        # send to listening blender script
        active_socket.sendall(str.encode(argstring))
    elif isinstance(argstring, np.ndarray):
        active_socket.send(argstring.tobytes())     # doesn't work on Blender side

    # get output from script that says we're done
    data = active_socket.recv(64)

    if verbose:
        print("Received", data.decode('utf-8'))
        print("Reading image from {}".format(os.path.join(shmpath, 'img{}.png'.format(values['id']))))

    if report_timing:
        print("Rendertime:", time.time()-start)

    return get_tensor(shmpath, values['id'], is_batched=is_batched, batchsize=batchsize).to(device)


def close_blender(active_socket):
    active_socket.close()
    print("Closed blender.")
