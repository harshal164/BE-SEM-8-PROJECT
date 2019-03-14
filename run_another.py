# -*- coding: utf-8 -*-

    valid_lr_img = get_imgs_fn('test.png', 'input/')  # if you want to test your own image
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
    tl.vis.save_image(out[0], 'output' + '/valid_gen.png')
    tl.vis.save_image(valid_lr_img, 'output' + '/valid_lr.png')
    tl.vis.save_image(valid_hr_img, 'output' + '/valid_hr.png')

    out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
    tl.vis.save_image(out_bicu, 'output' + '/valid_bicubic.png')

