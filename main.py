import distributed
import torch



if __name__ == '__main__':    

    dc = distributed.DistributionCenter(4)

    t = torch.ones((5,5))

    dc.shardTensor(t)

