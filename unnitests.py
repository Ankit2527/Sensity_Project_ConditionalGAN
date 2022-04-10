import unittest
import numpy as np
import torch
from network import Discriminator

class Test_Discriminator(unittest.TestCase):
    @torch.no_grad()
    def test_output_values(self):
        np.random.seed(0)
        torch.manual_seed(0)
        discriminator = Discriminator(img_dim = 784, output_dim = 1)
        batch_dim = 128
        z = torch.randn(batch_dim, 784)
        l = torch.randint(0, 9, (batch_dim,))
        preds = discriminator(z, l)
        self.assertTrue((preds > 0).any(),
                        msg = "As sigmoid is applied at the discriminator output, therefore the output values will only be positive " )

if __name__ == '__main__':
    test = unittest.TestLoader().loadTestsFromTestCase(Test_Discriminator)
    unittest.TextTestRunner().run(test)