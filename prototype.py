import imageio
import numpy as np
import math


def conform(val):
    if val < 0:
        val = 0
    elif val > 255:
        val = 255
    return val


class Window:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def convolve(self, channel, funcs):
        res = np.zeros_like(channel)
        half_height, half_width = self.height//2, self.width//2
        img = np.pad(
            channel,
            pad_width=max(half_width, half_height),
            mode='edge')
        for y in range(half_height, len(img)-half_height):
            for x in range(half_width, len(img[y])-half_width):
                weight, pcolor = 0, 0
                for k in range(-half_height, half_height+1):
                    for l in range(-half_width, half_width+1):
                        full = 1
                        for fun in funcs:
                            full *= fun(y, x, y - k, x - l, img)
                        weight += full
                        pcolor += full * img[y - k, x - l]
                res[y - half_height, x - half_width] = conform(pcolor / weight)
        return res


def gauss(x, s):
    return math.exp(-(x**2)/s)


def dist(y_1, x_1, y_2, x_2, img):
    sigma_dist = 15000
    dist = math.sqrt(math.pow(y_1 - y_2, 2) + math.pow(x_1 - x_2, 2))
    return gauss(dist, sigma_dist)


def color(y_1, x_1, y_2, x_2, img):
    sigma_color = 15000
    color = img[y_1, x_1] - img[y_2, x_2]
    return gauss(color, sigma_color)


if __name__ == "__main__":
    image = imageio.imread("example.png").astype(np.float)
    newimg = np.copy(image)

    funcs = [dist, color]
    w = Window(7, 7)
    for i in range(3):
        newimg[:, :, i] = w.convolve(image[:, :, i], funcs)
    imageio.imwrite("result.png", newimg.astype(np.uint8), format='png')
