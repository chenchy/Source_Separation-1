import torch.nn as nn

class transformer(nn.Module):
    def __init__(self, d_model=1024, nhead=8, num_encoder_layers=6,
    			 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
    			 activation='relu'):
    	
		transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)

	def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x