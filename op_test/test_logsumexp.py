import torch
import paddle
import paddle.fluid as fluid
import numpy as np
from datetime import datetime

import unittest


class CompareDotOp(unittest.TestCase):

    def _check_accuracy(self, dim, keepdim):
        for ind in range(1000):
            i = np.random.randint(3, 20, size=1)[0]
            j = np.random.randint(3, 20, size=1)[0]
            k = np.random.randint(3, 20, size=1)[0]
            np_x = np.random.uniform(0.1, 10, [i, j, k]).astype(np.float64)

            with fluid.dygraph.guard():
                paddle_x = fluid.dygraph.to_variable(np_x)
                paddle_res = paddle.logsumexp(paddle_x, dim, keepdim).numpy()

            torch_x = torch.from_numpy(np_x)
            torch_res = torch.logsumexp(torch_x, dim, keepdim)
            self.assertTrue(np.allclose(paddle_res, torch_res.numpy()))


    def test_api_accuracy(self):
        self._check_accuracy([0, 1, 2], False)

    def test_api_accuracy_1(self):
        self._check_accuracy(1, False)

    def test_api_accuracy_2(self):
        self._check_accuracy([0, 2], False)

    def test_api_accuracy_3(self):
        self._check_accuracy([0, 2], True)

    def test_api_efficiency(self, gpu=False):
        self._check_efficiency()
        self._check_efficiency(gpu=True)

    def _check_efficiency(self, loop=20000, gpu=False):
        total_paddle_time = 0
        total_torch_time = 0
        device = torch.device("cuda:0") if gpu else torch.device("cpu")
        place = fluid.CUDAPlace(0) if gpu else fluid.CPUPlace()

        for _ in range(loop):
            i = np.random.randint(3, 20, size=1)[0]
            j = np.random.randint(3, 20, size=1)[0]
            k = np.random.randint(3, 20, size=1)[0]
            np_x = np.random.uniform(0.1, 10, [i, j, k]).astype(np.float64)

            torch_x = torch.from_numpy(np_x).to(device)
            x = datetime.now()
            torch_res = torch.logsumexp(torch_x, (0, 1, 2))
            total_torch_time += (datetime.now() - x).total_seconds()

            with fluid.dygraph.guard(place):
                paddle_x = fluid.dygraph.to_variable(np_x)
                x = datetime.now()
                paddle_res = paddle.logsumexp(paddle_x).numpy()
                total_paddle_time += (datetime.now() - x).total_seconds()
         
        print(total_paddle_time, total_torch_time)   
    

if __name__ == "__main__":
    unittest.main()

