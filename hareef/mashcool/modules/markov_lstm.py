import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn import init



class LSTMCell(nn.Module):
    """A basic LSTM cell."""
    def __init__(self, input_size, hidden_size, k_cells, is_training=True, use_bias=True):
        super(LSTMCell, self).__init__()
        self.n = input_size
        self.m = hidden_size
        self.k = k_cells      # number of RNN cells
        self.is_training = is_training
        self.W_xz = nn.Parameter(torch.FloatTensor(self.n, self.k))  # affine transform x_t -> (z_1, z_2, ..., z_K)
        self.W_hz = nn.Parameter(torch.FloatTensor(self.m, self.k))  # affine transform   h -> (z_1, z_2, ..., z_K)

        self.W_x_4gates = nn.Parameter(torch.FloatTensor(self.n, 4 * self.m * self.k))
        self.W_h_4gates = nn.Parameter(torch.FloatTensor(self.m, 4 * self.m * self.k))

        self.use_bias = use_bias
        if use_bias:
            self.b_xz = nn.Parameter(torch.FloatTensor(self.k))  # size = (1, k)
            # self.b_hz = nn.Parameter(torch.FloatTensor(self.k))  # size = (1, k)
            self.b_4gates = nn.Parameter(torch.FloatTensor(4 * self.m * self.k))   # size = (1, 4 * m * k)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        print("LSTM cell parameters reseted!")


    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.W_xz.data.size(1))
        self.W_xz.data.uniform_(-stdv1, stdv1)

        stdv2 = 1. / math.sqrt(self.W_hz.data.size(1))
        self.W_hz.data.uniform_(-stdv2, stdv2)

        stdv3 = 1. / math.sqrt(self.W_x_4gates.data.size(1))
        self.W_x_4gates.data.uniform_(-stdv3, stdv3)

        stdv4 = 1. / math.sqrt(self.W_h_4gates.data.size(1))
        self.W_h_4gates.data.uniform_(-stdv4, stdv4)

        if self.b_xz is not None:
            self.b_xz.data.uniform_(-stdv1, stdv1)

        if self.b_4gates is not None:
            self.b_4gates.data.uniform_(-stdv4, stdv4)

    def forward(self, x_t, tau, s_t, is_training):
        """ Args:
            x_t: input at time step t = (batch, input_size) tensor containing input features.
            tau: annealing temperature for Gumbel softmax
            s_t: state at time step t (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
            is_training: indicating status of training or not.
            Returns:
            state = (h_t, c_t): Tensors containing the next hidden and cell state. """

        h_0, c_0 = s_t   # state s_t = (h_t, c_t) with  size h_0 = size c_0 = (N, m)
        N = h_0.size(0)  # batch size
        self.tau = tau
        self.is_training = is_training

        # expand (repeat) bias for batch processing
        batch_b_xz = (self.b_xz.unsqueeze(0).expand(N, *self.b_xz.size()))    # size = (N, k)
        # batch_b_hz = (self.b_hz.unsqueeze(0).expand(N, *self.b_hz.size()))  # size = (N, k)
        batch_b_4gates = (self.b_4gates.unsqueeze(0).expand(N, *self.b_4gates.size()))  # size = (N, 4* m* k)

        ''' logit encoder: logit_z = (W1 * x_t + b1) + (W2 * h_{t-1} + b2) in R^K '''
        logit_z = torch.addmm(batch_b_xz, x_t, self.W_xz) + torch.mm(h_0, self.W_hz)

        # probability q_z(x_t, h_{t-1}) in R^K
        q_z = F.softmax(logit_z, dim=1)

        if self.is_training is True:
            z = F.gumbel_softmax(logit_z, self.tau, hard=False, eps=1e-10)
        else:
            if q_z.is_cuda:
                z = torch.cuda.FloatTensor(N, self.k).zero_()      # create a GPU zero tensor for 1-hot
            else:
                z = torch.FloatTensor(N, self.k).zero_()           # create a CPU zero tensor for 1-hot

            z.scatter_(1, torch.max(q_z, dim=1)[1].view(N, 1), 1) # find which position is max

        # k LSTM's part
        A = torch.mm(x_t, self.W_x_4gates)                      # (W_x_4gates * x + b_4gates)  , output dim = 4* m* k
        B = torch.addmm(batch_b_4gates, h_0, self.W_h_4gates)   # (W_h_4gates * h_0 + b_4gates), output dim = 4* m* k
        ''' this step B is a bit strange? because it may breakdown the independence of the k LSTMs'''

        # 4 gates in LSTM
        I_gate, G_gate, F_gate, O_gate = torch.split(A + B, split_size_or_sections=(self.m * self.k), dim=1)

        # Expand c_0 -> C_0 has dim=k
        C_0 = c_0.repeat(1, self.k)

        '''for K LSTMs: C_t = (c^1_t, c^2_t, .... , c^K_t )  
                        H_t = (h^1_t, h^2_t, .... , h^K_t ) [vectorized version for K RNNs] '''
        C_t = torch.mul(torch.sigmoid(F_gate), C_0) + torch.mul(torch.sigmoid(I_gate), torch.tanh(G_gate))  # C_t = new c of K dim
        H_t = torch.mul(torch.sigmoid(O_gate), torch.tanh(C_t))                                         # H_t = new h of K dim

        # reshape C_t & H_t has dim = (N, k, m)   (hidden_size = m)
        C_t = C_t.view([N, self.k, self.m])
        H_t = H_t.view([N, self.k, self.m])

        # sum over K LSTMs (like K ensemble) to become 1 LSTM cell & hidden state: (h_t , c_t)
        h_t = torch.einsum('nkm,nk->nm', (H_t, z))    # size= (batch, output-dim), no time
        c_t = torch.einsum('nkm,nk->nm', (C_t, z))    # size= (batch, output-dim), no time

        return z, q_z, h_t, c_t

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class LSTM(nn.Module):
    """A module that runs multiple steps of LSTM."""
    def __init__(self, cell_class, input_size, hidden_size, output_size,
                 num_layers=1, k_cells=2, use_bias=True, dropout_prob=0.):
        super(LSTM, self).__init__()
        self.cell_class = cell_class
        self.n = input_size
        self.m = hidden_size
        self.k = k_cells      # number of RNN cells
        self.output_size = output_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.dropout_prob = dropout_prob

        self.fc = nn.Linear(self.m, self.output_size)  # output = CNN embedding latent variables

        for layer in range(num_layers):
            layer_input_size = self.n if layer == 0 else self.m
            cell = cell_class(input_size=layer_input_size, hidden_size=self.m, k_cells=self.k, use_bias=self.use_bias)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout_prob)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, x, tau, length, s_t, is_training):
        T = x.size(1)  # time is the second dimension

        z_T = []  # z_T = (z_1, z_2, .... , ) collecting all z's over time
        qz_T = []
        h_T = []  # h_T = (h_1, h_2, .... , ) collecting all hidden state h_t over time
        S_T = []  # S_T = [ (h_1, c_1), (h_2, c_2), .... , ] collecting all (h_t, c_t) over time
        # output = []
        for t in range(T):
            # one time step
            z, q_z, h_t, c_t = cell(x[:, t, :], tau, s_t, is_training)

            # to bound time steps of time sequences
            time_mask = (t < length).float().unsqueeze(1).expand_as(h_t)
            h_t, c_t = h_t * time_mask + s_t[0] * (1 - time_mask), c_t * time_mask + s_t[1] * (1 - time_mask)
            s_t = (h_t, c_t)     # state t = (h_t, c_t)

            h_T.append(h_t)
            z_T.append(z)
            qz_T.append(q_z)

        z_T = torch.stack(z_T, 0).transpose_(0, 1)    # [transpose to batch first], size=(N, T, output-dim)
        qz_T = torch.stack(qz_T, 0).transpose_(0, 1)  # [transpose to batch first], size=(N, T, output-dim)
        h_T = torch.stack(h_T, 0).transpose_(0, 1)    # [transpose to batch first], size=(N, T, output-dim)

        return z_T, qz_T, h_T, s_t

    def forward(self, x, tau, length=None, s_t=None, is_training=True):

        # batch is assumed first dimension of input x
        N, T, n = x.size()
        ''' N = batch size, T = total time steps, n = input feature dimension '''

        # RNN input temporal length limit
        if length is None:
            length = Variable(torch.LongTensor([T] * N))
            if x.is_cuda:
                device = x.get_device()
                length = length.cuda(device)
        if s_t is None:
            # put an initialization
            s_t = Variable(x.data.new(N, self.m).zero_())
            s_t = (s_t, s_t)

        all_layer_h_t = []
        all_layer_c_t = []
        layer_h_T = None

        # creating depth of LSTMs
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)   # get the cell of certain layer
            layer_z_T, layer_qz_T, layer_h_T, (layer_h_t, layer_c_t) = LSTM._forward_rnn(cell, x, tau,
                                                                                         length, s_t, is_training)

            ''' x=data input if layer=0; x=hidden units if layer > 0 '''
            x = self.dropout_layer(layer_h_T)
            all_layer_h_t.append(layer_h_t)
            all_layer_c_t.append(layer_c_t)

        all_layer_h_t = torch.stack(all_layer_h_t, 0)
        all_layer_c_t = torch.stack(all_layer_c_t, 0)

        output = self.fc(layer_h_t)

        return output, layer_z_T, layer_qz_T, layer_h_T, (all_layer_h_t, all_layer_c_t)

