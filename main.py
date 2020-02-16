




'''
main.py
'''





import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os


import selector





class Train:

    def __init__(self, model, lossfunc,
                 bpsize, beta, sample_min, max_len=3000,
                 lr=0.01, momentum=0.9, weight_decay=0.0001):
        self.net = self.choose_net(model)
        self.criterion = self.choose_lossfunc(lossfunc)
        self.opti = torch.optim.SGD(self.net.parameters(),
                                    lr=lr, momentum=momentum,
                                    weight_decay=weight_decay)
        self.selector = selector.Selector(bpsize, beta,
                                          sample_min, max_len)
        self.gpu()
        self.generate_path()
        self.acc_rates = []
        self.errors = []

    def choose_net(self, model):
        net = getattr(
            torchvision.models,
            model,
            None
        )
        if net is None:
            raise ValueError("no such model")
        return net()

    def choose_lossfunc(self, lossfunc):
        lossfunc = getattr(
            nn,
            lossfunc,
            None
        )
        if lossfunc is None:
            raise ValueError("no such lossfunc")
        return lossfunc



    def gpu(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("Let'us use %d GPUs" % torch.cuda.device_count())
            self.net = nn.DataParallel(self.net)
        self.net = self.net.to(self.device)



    def generate_path(self):
        """
        生成保存数据的路径
        :return:
        """
        try:
            os.makedirs('./paras')
            os.makedirs('./logs')
            os.makedirs('./infos')
        except FileExistsError as e:
            pass
        name = self.net.__class__.__name__
        paras = os.listdir('./paras')
        logs = os.listdir('./logs')
        infos = os.listdir('./infos')
        number = max((len(paras), len(logs), len(infos)))
        self.para_path = "./paras/{0}{1}.pt".format(
            name,
            number
        )

        self.log_path = "./logs/{0}{1}.txt".format(
            name,
            number
        )
        self.info_path = "./infos/{0}{1}.npy".format(
            name,
            number
        )


    def log(self, strings):
        """
        运行日志
        :param strings:
        :return:
        """
        # a 往后添加内容
        with open(self.log_path, 'a', encoding='utf8') as f:
            f.write(strings)

    def save(self):
        """
        保存网络参数
        :return:
        """
        torch.save(self.net.state_dict(), self.para_path)

    def derease_lr(self, multi=0.96):
        """
        降低学习率
        :param multi:
        :return:
        """
        self.opti.param_groups[0]['lr'] *= multi


    def train(self, trainloder, epochs=50):
        data_size = len(trainloder) * trainloder.batch_size
        part = int(trainloder.batch_size / 2)
        for epoch in range(epochs):
            running_loss = 0.
            total_loss = 0.
            acc_count = 0.
            if (epoch + 1) % 8 is 0:
                self.derease_lr()
                self.log(#日志记录
                    "learning rate change!!!\n"
                )
            for i, data in enumerate(trainloder):
                imgs, labels = data
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                out = self.net(imgs)
                _, pre = torch.max(out, 1)  #判断是否判断正确
                acc_count += (pre == labels).sum().item() #加总对的个数

                losses = (
                    self.criterion(out[i], labels[i])
                    for i in range(len(labels))
                )

                self.opti.zero_grad()
                self.selector(losses) #选择
                self.opti.step()

                running_loss += sum(losses).item()

                if (i+1) % part is 0:
                    strings = "epoch {0:<3} part {1:<5} loss: {2:<.7f}\n".format(
                        epoch, i, running_loss / part
                    )
                    self.log(strings)#日志记录
                    total_loss += running_loss
                    running_loss = 0.
            self.acc_rates.append(acc_count / data_size)
            self.errors.append(total_loss / data_size)
            self.log( #日志记录
                "Accuracy of the network on %d train images: %d %%\n" %(
                    data_size, acc_count / data_size * 100
                )
            )
            self.save() #保存网络参数
        #保存一些信息画图用
        np.save(self.info_path, {
            'acc_rates': np.array(self.acc_rates),
            'errors': np.array(self.errors)
        })




if __name__ == "__main__":

    import OptInput
    args = OptInput.Opi()
    args.add_opt(command="model", default="resnet34")
    args.add_opt(command="lossfunc", default="CrossEntropyLoss")
    args.add_opt(command="bpsize", default=32)
    args.add_opt(command="beta", default=0.9)
    args.add_opt(command="sample_min", default=0.3)
    args.add_opt(command="max_len", default=3000)
    args.add_opt(command="lr", default=0.001)
    args.add_opt(command="momentum", default=0.9)
    args.add_opt(command="weight_decay", default=0.0001)

    args.acquire()

    root = "C:/Users/pkavs/1jupiterdata/data"

    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                          download=False,
                                          transform=transforms.Compose(
                                              [transforms.Resize(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                          ))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=8,
                                               pin_memory=False)



    dog = Train(**args.infos)
    dog.train(train_loader, epochs=1000)



























