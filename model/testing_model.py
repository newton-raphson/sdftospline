import unittest
from mapping import EncoderDecoder
import torch
from torchviz import make_dot
# from ..utils import nnutils
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
class TestModel(unittest.TestCase):
    def test_model(self):
        # Example usage:
        input_size = ( 8, 2)  # Single-channel images of size 256x256
        output_size = (1, 128, 128)  # Adjust based on your requirements
        num_hidden_layers = 3
        latent_dim = 10
        # Create an instance of the model
        model = EncoderDecoder(input_size)
        print(count_parameters(model))
        # print(model)
        test_input = torch.rand(1,8,2)
        op_shape = model(test_input).shape 
        # Generate the visualization
        output = model(test_input)
        graph = make_dot(output, params=dict(model.named_parameters()))
        graph.render(filename='encoder_decoder', format='png', cleanup=True)
        self.assertEqual(op_shape,torch.Size([1, 1, 128, 128]))
# verifyin if the model works correctly or not 
if __name__ == '__main__':
    unittest.main()