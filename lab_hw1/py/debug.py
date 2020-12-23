import torch
import torch.nn as nn
import os

class Conv2d_dump(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: int = 1,
                 padding: int = 0
                 ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.__dbg = False

    def set_debug(self, b):
        self.__dbg = b

    def forward(self, x):
        y = super().forward(x)
        if self.__dbg:
            cpu_x = x.to(torch.device("cpu"))
            cpu_y = y.to(torch.device("cpu"))
            cpu_w = self.weight.to(torch.device("cpu"))
            cpu_b = self.bias.to(torch.device("cpu"))
            for i in range(len(cpu_x)):
                dump_conv2d(cpu_x[i], cpu_y[i], cpu_w, cpu_b,
                            self.in_channels, self.out_channels, self.kernel_size,
                            self.stride, self.padding, i)
        return y

def dump_conv2d(x, y, weight, bias,
                in_channels, out_channels, kernel_size, stride, padding, iter):
    os.makedirs(f"dump/{os.getpid()}", exist_ok=True)
    with open(f"dump/{os.getpid()}/{iter}.txt", "w") as out:
        out.write("Image:\n")
        out.write(dumpN(x, (",\n\n", ",\n", ", ")))
        out.write("\n")
        out.write("Filter:\n")
        out.write(dumpN(weight, (",\n\n" + "#" * 16 + "\n\n", ",\n\n", ",\n", ", ")))
        out.write("\n")
        out.write("Bias:\n")
        out.write(dumpN(bias, (", ",)))
        out.write("\n")
        out.write("Feature map:\n")
        out.write(dumpN(y, (",\n\n", ",\n", ", ")))
        out.write("\n")

def dumpN(image, delimiter):
    assert image.dim() == len(delimiter)
    if image.dim() == 0:
        return f"{float(image):.4f}"
    return delimiter[0].join(map(lambda x: dumpN(x, delimiter[1:]), image))
