import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '7'
from Model.DCRNN.dcrnn_cell import DCGRUCell

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, adj_mx, config):
        self.adj_mx = adj_mx
        self.max_diffusion_step = config['transfer']['backbone']['max_diffusion_step']
        self.cl_decay_steps = config['transfer']['backbone']['cl_decay_steps']
        self.filter_type = config['transfer']['backbone']['filter_type']
        self.num_nodes = adj_mx.shape[0]
        self.num_rnn_layers = config['transfer']['backbone']['num_rnn_layers']
        self.rnn_units = config['transfer']['backbone']['rnn_units']
        self.hidden_state_size = self.num_nodes * self.rnn_units
        self.device = torch.device(config['basic']['device'])

class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj, config):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj, config)
        self.input_dim = config['transfer']['inp_dim']
        self.seq_len = config['transfer']['seq_len']
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.device, self.rnn_units, adj, self.max_diffusion_step, self.num_nodes,
                        filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=self.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)



class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, config):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, config)
        self.output_dim = config['transfer']['oup_dim']
        self.horizon = config['transfer']['pre_len']  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.device, self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None):
        """
        Decoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)



class DCRNNModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj, config):
        super().__init__()
        Seq2SeqAttrs.__init__(self, adj, config)
        self.encoder_model = EncoderModel(adj, config).to(config['basic']['device'])
        self.decoder_model = DecoderModel(adj, config).to(config['basic']['device'])
        self.linear = nn.Linear(config['transfer']['seq_len'], config['transfer']['pre_len'])
        self.cl_decay_steps = config['transfer']['backbone']['cl_decay_steps']
        self.use_curriculum_learning = config['transfer']['backbone']['use_curriculum_learning']

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))


    def encoder(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        # _, encoder_hidden_state = self.encoder_model(inputs, encoder_hidden_state)
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[:,:,t], encoder_hidden_state)

        return encoder_hidden_state

    
    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim),
                                device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        encoder_hidden_state = self.encoder(inputs) # [inp_dim, bz, nd*hid_sz]
        # self._logger.debug("Encoder complete, starting decoder")
        # print("Encoder complete, starting decoder")
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)
        # self._logger.debug("Decoder complete")
        # print("Decoder complete")
        # outputs = self.linear(outputs.permute(1,2,0))
        if batches_seen == 0:
            print(
                "Total trainable parameters {}".format(count_parameters(self))
            )
        return outputs

class RegionTrans(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj, args):
        super().__init__()
        Seq2SeqAttrs.__init__(self, adj, args)
        self.encoder_model = EncoderModel(adj, args).to(torch.device('cuda:'+ args.gpu_id))
        self.decoder_model = DecoderModel(adj, args).to(torch.device('cuda:'+ args.gpu_id))
        self.cl_decay_steps = args.cl_decay_steps
        self.use_curriculum_learning = args.use_curriculum_learning
        # TODO
        # self.linear = nn.Linear()

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))


    def encoder(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)

        return encoder_hidden_state

    
    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim),
                                device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        encoder_hidden_state = self.encoder(inputs)
        # self._logger.debug("Encoder complete, starting decoder")
        # print("Encoder complete, starting decoder")
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)
        # self._logger.debug("Decoder complete")
        # print("Decoder complete")
        if batches_seen == 0:
            print(
                "Total trainable parameters {}".format(count_parameters(self))
            )
        return torch.reshape(encoder_hidden_state.permute(1,0,2), (-1, self.num_nodes, self.rnn_units)), outputs.permute(1,2,0)


class MetaST(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj, args,STmem_num = 20):
        super().__init__()
        Seq2SeqAttrs.__init__(self, adj, args)
        self.encoder_model = EncoderModel(adj, args).to(torch.device('cuda:'+ args.gpu_id))
        self.decoder_model = DecoderModel(adj, args).to(torch.device('cuda:'+ args.gpu_id))
        # self.linear = nn.Linear()
        self.key_query = nn.Linear(self.rnn_units, self.rnn_units).to(torch.device('cuda:'+ args.gpu_id))
        # self.hidden_state_size = self.rnn_units
        self.mem_embed = nn.Parameter(torch.empty(STmem_num,self.rnn_units),requires_grad=True).to(torch.device('cuda:'+ args.gpu_id))
        self.mem_query = nn.Linear(self.rnn_units, self.rnn_units).to(torch.device('cuda:'+ args.gpu_id))
        self.mem_value = nn.Linear(self.rnn_units, self.rnn_units).to(torch.device('cuda:'+ args.gpu_id))
        self.softmax = nn.Softmax(dim=1)
        nn.init.uniform_(self.mem_embed)
        self.cl_decay_steps = args.cl_decay_steps
        self.use_curriculum_learning = args.use_curriculum_learning


    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))


    def encoder(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)

        return encoder_hidden_state

    
    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim),
                                device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, labels=None, batches_seen=None):
        """
        use ST mem & attention
        """
        encoder_hidden_state = self.encoder(inputs)
        encoder_hidden = torch.reshape(encoder_hidden_state.permute(1,0,2), (-1, self.num_nodes, self.rnn_units))
        key = self.key_query(encoder_hidden)
        mem_attention = self.softmax(torch.matmul(key, torch.t(self.mem_query(self.mem_embed))))
        encoder_hidden_state = torch.matmul(mem_attention, self.mem_value(self.mem_embed))
        encoder_hidden_state = encoder_hidden_state.reshape(1,inputs.size()[1],-1)
        # print(encoder_hidden_state)
        # self._logger.debug("Encoder complete, starting decoder")
        # print("Encoder complete, starting decoder")
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)
        # self._logger.debug("Decoder complete")
        # print("Decoder complete")

        if batches_seen == 0:
            print(
                "Total trainable parameters {}".format(count_parameters(self))
            )
        return outputs