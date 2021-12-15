import torch
import h5py
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.models as models
from tqdm import tqdm

import config
from data_utils.vivqa_image import ViVQAImages
from data_utils.utils import get_transform


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # self.model = caffe_resnet.resnet152(pretrained=True)
        self.model = models.resnet152(pretrained=True)

        def save_output(module, input, output):
            self.buffer = output

        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer


def create_vivqa_loader(path):
    transform = get_transform(config.image_size)
    dataset = ViVQAImages(path, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.preprocess_batch_size,
        num_workers=config.data_workers,
        shuffle=False,
        pin_memory=True,
    )
    return data_loader


def main():
    cudnn.benchmark = True

    net = ResNet().cuda()
    net.eval()

    train_loader = create_vivqa_loader(config.train_path)
    test_loader = create_vivqa_loader(config.test_path)
    features_shape = (
        len(train_loader.dataset) + len(test_loader.dataset),
        config.output_features,
        config.output_size,
        config.output_size
    )

    with h5py.File(config.preprocessed_path, 'w', libver='latest') as fd:
        features = fd.create_dataset('features', shape=features_shape, dtype='float16')
        coco_ids = fd.create_dataset('ids', shape=(len(train_loader.dataset) + len(test_loader.dataset), ), dtype='int32')

        i = j = 0
        for ids, imgs in tqdm(train_loader):
            imgs = imgs.clone().cuda()
            out = net(imgs)

            j = i + imgs.size(0)
            features[i:j, :, :] = out.data.cpu().numpy().astype('float16')
            coco_ids[i:j] = ids.numpy().astype('int32')
            i = j

        for ids, imgs in tqdm(test_loader):
            imgs = imgs.clone().cuda()
            out = net(imgs)

            j = i + imgs.size(0)
            features[i:j, :, :] = out.data.cpu().numpy().astype('float16')
            coco_ids[i:j] = ids.numpy().astype('int32')
            i = j


if __name__ == '__main__':
    main()
