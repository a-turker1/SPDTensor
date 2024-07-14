import distributed
import torch



if __name__ == '__main__':    

    dc = distributed.DistributionCenter(4)

    t = torch.ones((8,5))

    stensor1 = dc.shardTensor(t)
    stensor2 = dc.replicateTensor(t*2)
    stensor3 = dc.replicateTensor(t*5)
    res = stensor1 + 5
    res2 = res + stensor3


