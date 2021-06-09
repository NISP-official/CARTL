import torch

from src.networks import make_blocks
from src.networks import (wrn34_10, parseval_retrain_wrn34_10, parseval_resnet18,
                          wrn, parseval_wrn, resnet18, resnet, parseval_resnet,
                          WideResNet, ParsevalWideResNet)
from src.networks.utils import WRNBlocks

def test_wrn_make_blocks():
    model = wrn34_10(num_classes=10)
    blocks = make_blocks(model)

    assert blocks.get_total_blocks() == 17
    assert isinstance(blocks.get_block(17)[0], torch.nn.Linear)
    assert isinstance(blocks.get_block(16)[0], torch.nn.BatchNorm2d)
    for i in range(1, 16):
        assert isinstance(blocks.get_block(14), wrn.BasicBlock)

    model = parseval_retrain_wrn34_10(k=8, num_classes=10)
    blocks = make_blocks(model)

    assert blocks.get_total_blocks() == 17
    assert isinstance(blocks.get_block(17)[0], torch.nn.Linear)
    assert isinstance(blocks.get_block(16)[0], torch.nn.BatchNorm2d)
    for i in range(10, 16):
        assert isinstance(blocks.get_block(i), parseval_wrn.ParsevalBasicBlock)
    for i in range(1, 10):
        assert isinstance(blocks.get_block(i), wrn.BasicBlock)
    
def test_make_wrnblocks(k=6):
    for n in [5, 4]:
        model = WideResNet(4 + n * 6, num_classes=10, widen_factor=10, dropRate=0)
        blocks = WRNBlocks(model)

        assert blocks.get_total_blocks() == 2 + 3 * n
        assert isinstance(blocks.get_block(blocks.get_total_blocks())[0], torch.nn.Linear)
        assert isinstance(blocks.get_block(blocks.get_total_blocks() - 1)[0], torch.nn.BatchNorm2d)
        for i in range(1, 3 * n + 1):
            assert blocks.get_block(i) == getattr(blocks, f"block{i}")
            assert isinstance(blocks.get_block(i), wrn.BasicBlock)
            assert blocks.get_block(i).conv1.out_channels == 160 * int(2 **((i - 1) // n))
            assert blocks.get_block(i).conv2.in_channels == 160 * int(2 **((i - 1) // n))
        
        model = ParsevalWideResNet(k, 4 + n * 6, num_classes=10, widen_factor=10, dropRate=0)
        blocks = WRNBlocks(model)

        assert blocks.get_total_blocks() == 2 + 3 * n
        assert isinstance(blocks.get_block(blocks.get_total_blocks())[0], torch.nn.Linear)
        assert isinstance(blocks.get_block(blocks.get_total_blocks() - 1)[0], torch.nn.BatchNorm2d)
        for i in range(1, blocks.get_total_blocks() + 1 - k):
            assert isinstance(blocks.get_block(i), wrn.BasicBlock)
        for i in range(blocks.get_total_blocks() + 1 - k, 3 * n + 1):
            assert isinstance(blocks.get_block(i), parseval_wrn.ParsevalBasicBlock)
        
    print("Test Passed")

def test_resnet_make_blocks():
    model = resnet18(num_classes=10)
    blocks = make_blocks(model)

    assert blocks.get_total_blocks() == 9
    assert isinstance(blocks.get_block(9)[0], torch.nn.Linear)
    for i in range(1, 9):
        assert isinstance(blocks.get_block(i), resnet.BasicBlock)

    model = parseval_resnet18(k=3, num_classes=10)
    blocks = make_blocks(model)

    assert blocks.get_total_blocks() == 9
    assert isinstance(blocks.get_block(9)[0], torch.nn.Linear)
    for i in range(7, 9):
        assert isinstance(blocks.get_block(i), parseval_resnet.ParsevalBasicBlock)
    for i in range(1, 7):
        assert isinstance(blocks.get_block(i), resnet.BasicBlock)
