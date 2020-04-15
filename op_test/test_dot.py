import torch
import paddle
import paddle.fluid as fluid
import numpy as np
from datetime import datetime

import unittest


class CompareDotOp(unittest.TestCase):

    def test_api_accuracy(self):
        for i in range(10000):
            n = np.random.randint(100, 1000, size=1)[0]
            np_x = np.random.uniform(0.1, 10, [n]).astype(np.float64)
            np_y = np.random.uniform(0.1, 10, [n]).astype(np.float64)

            with fluid.dygraph.guard():
                paddle_x = fluid.dygraph.to_variable(np_x)
                paddle_y = fluid.dygraph.to_variable(np_y)
                paddle_res = paddle.dot(paddle_x, paddle_y).numpy()

            torch_x = torch.from_numpy(np_x)
            torch_y = torch.from_numpy(np_y)
            torch_res = torch.dot(torch_x, torch_y)
            self.assertTrue(np.allclose(paddle_res[0], torch_res.numpy()))

    def test_api_efficiency(self, gpu=False):
        self.compare_paddle_and_torch()
        self.compare_paddle_and_torch(gpu=True)

    def compare_paddle_and_torch(self, loop=20000, gpu=False):
        total_paddle_time = 0
        total_torch_time = 0
        device = torch.device("cuda:0") if gpu else torch.device("cpu")
        place = fluid.CUDAPlace(0) if gpu else fluid.CPUPlace()

        for i in range(loop):
            n = np.random.randint(100, 1000, size=1)[0]
            np_x = np.random.uniform(0.1, 10, [n]).astype(np.float64)
            np_y = np.random.uniform(0.1, 10, [n]).astype(np.float64)

            torch_x = torch.from_numpy(np_x).to(device)
            torch_y = torch.from_numpy(np_y).to(device)
            x = datetime.now()
            torch_res = torch.dot(torch_x, torch_y)
            total_torch_time += (datetime.now() - x).total_seconds()

            with fluid.dygraph.guard(place):
                paddle_x = fluid.dygraph.to_variable(np_x)
                paddle_y = fluid.dygraph.to_variable(np_y)
                x = datetime.now()
                paddle_res = paddle.dot(paddle_x, paddle_y).numpy()
                total_paddle_time += (datetime.now() - x).total_seconds()

         
        print(total_paddle_time, total_torch_time)   
    


if __name__ == "__main__":
    unittest.main()

