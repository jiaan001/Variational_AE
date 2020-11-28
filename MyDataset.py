from torch.utils.data import Dataset
from PIL import Image
import os

# 过滤警告信息
import warnings

warnings.filterwarnings("ignore")


class MyDataset(Dataset):  # 继承Dataset
    def __init__(self, path_dir, transform):  # 初始化一些属性
        self.path_dir = path_dir  # 文件路径
        self.transform = transform  # 对图形进行处理，如标准化、截取、转换等
        self.images = os.listdir(self.path_dir)  # 把路径下的所有文件放在一个列表中

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  # 根据索引index返回图像
        image_index = self.images[index]  # 根据索引获取图像文件名称
        img_path = os.path.join(self.path_dir, image_index)  # 获取图像的路径或目录
        img = Image.open(img_path).convert('RGB')  # 读取图像

        if self.transform is not None:
            img = self.transform(img)
        return img

if __name__ == '__main__':
    dataset = MyDataset('student/train/', transform=None)
    img = dataset[0]  # 方法__getitem__(0)


