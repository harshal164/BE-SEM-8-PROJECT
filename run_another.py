
from model import *
from utils import *
from config import config, log_config


batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1

n_epoch_init = config.TRAIN.n_epoch_init

n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))



dir = 'input/'
data_files=[]
data_files += [file for file in os.listdir(dir) if os.path.isfile(dir + file)]



for inp_image in data_files:
    
    # -*- coding: utf-8 -*-
    
    valid_lr_img = get_imgs_fn(inp_image, 'input/')  # if you want to test your own image
    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
    # print(valid_lr_img.min(), valid_lr_img.max())
    
    size = valid_lr_img.shape
    # t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image') # the old version of TL need to specify the image size
    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')
    
    net_g = SRGAN_g(t_image, is_train=False, reuse=tf.AUTO_REUSE)
    
    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name= 'g_srgan.npz', network=net_g)
    
    ###======================= EVALUATION =============================###
    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
    print("took: %4.4fs" % (time.time() - start_time))
    
    print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    tl.vis.save_image(out[0], 'output/' +inp_image[:-4]+' HD.png')
    tl.vis.save_image(valid_lr_img, 'output/' +inp_image[:-4]+' LD.png')
    
    
    out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
    tl.vis.save_image(out_bicu, 'output/' + inp_image[:-4]+' bicubic.png')
    
