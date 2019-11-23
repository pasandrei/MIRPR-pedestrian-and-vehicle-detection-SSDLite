import torch
import numpy

def test():
    x = torch.rand(1,4 * 6, 4, 4)
    batch_size, channels, H, W = x.size()

    x = x.permute(0, 2, 3, 1).contiguous()  # B x H x W x (4*k)

    # batch, H*W*k, #classes or 4 (bbox coords)
    rez = x.view(batch_size, -1, channels//6)

    # eliminate batch
    x, rez = x[0], rez[0]
    print(x.shape, rez.shape)

    # im checking to see if anchors of the first two cells in the feature map correspond
    # to the first 12 lines of the anchors
    print("Predictions of top left feature map: ", x[0][0])
    print("First six elements of unpacked tensor: ", rez[:6])

    print("Predictions of next to top left feature map: ", x[0][1])
    print("Next six elements of unpacked tensor: ", rez[6:12])

    # assertions
    # unpack rez
    rez1 = rez[:6].view(24)
    assert torch.all(torch.eq(rez1, x[0][0])).item() is True

    rez2 = rez[6:12].view(24)
    assert torch.all(torch.eq(rez2, x[0][1])).item() is True

    # or written with numpy if you wish
    rez, x = rez.numpy(), x.numpy()
    rez1 = numpy.reshape(rez[:6], 24, 'C')
    assert numpy.array_equal(rez1, x[0][0]) is True

    rez2 = numpy.reshape(rez[6:12], 24, 'C')
    assert numpy.array_equal(rez2, x[0][1]) is True

    print('Model output tensor unpacking 100 % C row major order')
