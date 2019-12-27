# coding=utf-8
import os

__author__ = "max"


def download_model(model_name):
    pretrained_path = "./pretrained-model"
    os.makedirs(pretrained_path, exist_ok=True)

    model_name = model_name + "_pretrained"
    print("Download pretrained", model_name)
    if not os.path.exists("pretrained-model/{}".format(model_name)):
        os.system(
            "cd {} && wget https://paddle-imagenet-models-name.bj.bcebos.com/{}.tar -O {}.tar && tar -xvf {}.tar".format(
                pretrained_path, model_name, model_name, model_name))
        # remove fc layers
        os.system("cd {}/{} && rm -rf *fc*".format(pretrained_path, model_name))


def build_basic_net(model_net, is_train, use_label_somoothing):
    logger.info("Building network {}, class_dim:{}, use_label_smoothing: {}".format(ModelNet, class_dim, use_label_smoothing))
    image = fluid.layers.data(name='img', shape=train_parameters['input_size'], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    model = ModelNet()
    net_out = model.net(input=image, class_dim=class_dim)
    softmax_out = fluid.layers.softmax(net_out, use_cudnn=True)

    if is_train and use_label_smoothing:
        label_one_hot = fluid.layers.one_hot(input=label, depth=class_dim)
        smooth_label = fluid.layers.label_smooth(label=label_one_hot, epsilon=label_smoothing_epsilon, dtype="float32")
        loss = fluid.layers.cross_entropy(input=softmax_out, label=smooth_label, soft_label=True)
    else:
        loss = fluid.layers.cross_entropy(input=softmax_out, label=label)

    avg_loss = fluid.layers.mean(loss)
    acc_top1 = fluid.layers.accuracy(input=softmax_out, label=label, k=1)
    # acc_top5 = fluid.layers.accuracy(input=softmax_out, label=label, k=5)
    return image, label, softmax_out, avg_loss, acc_top1
