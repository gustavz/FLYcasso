"""Persistent full-connectome dynamics with sparse ports and per-edge learning.

Ports and anatomy are immutable buffers. No dense neuron projection or bypass.
Rate dynamics and fitted synaptic efficacy are modeling choices, not physiology.
"""
import numpy as np
import torch
from torch import nn
from flycasso.model import load_graph


class EdgeMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, values, ptr, idx, rows, tptr, tidx, transposed):
        ctx.save_for_backward(x, values, ptr, idx, rows, tptr, tidx, transposed)
        if x.device.type == "mps":
            from flycasso.metal import multiply
            return multiply(x, ptr, idx, values)
        w = torch.sparse_csr_tensor(ptr, idx, values, size=(len(ptr)-1, len(ptr)-1))
        return torch.sparse.mm(w, x)

    @staticmethod
    def backward(ctx, grad):
        x, values, ptr, idx, rows, tptr, tidx, transposed = ctx.saved_tensors
        if x.device.type == "mps":
            from flycasso.metal import multiply, edge_gradient
            dx = multiply(grad, tptr, tidx, transposed) if ctx.needs_input_grad[0] else None
            dv = edge_gradient(x, grad, rows, idx) if ctx.needs_input_grad[1] else None
        else:
            wt = torch.sparse_csr_tensor(tptr, tidx, transposed, size=(len(ptr)-1, len(ptr)-1))
            dx = torch.sparse.mm(wt, grad) if ctx.needs_input_grad[0] else None
            dv = torch.empty_like(values) if ctx.needs_input_grad[1] else None
            if dv is not None:
                for start in range(0, len(idx), 65536):
                    s = slice(start, start+65536)
                    dv[s] = (grad[rows[s].long()] * x[idx[s].long()]).sum(1)
        return dx, dv, None, None, None, None, None, None


class Circuit(nn.Module):
    def __init__(self, graph, ports, control="real", seed=42, calibration=None):
        super().__init__()
        w, self.metadata = load_graph(graph, control, seed)
        n = w.size(0)
        if set(ports)!={'input_channel','input_scale','output_index','output_scale'}:
            raise ValueError('Expected input and output port indices and scales')
        if ports['input_channel'].shape!=(n,) or ports['input_scale'].shape!=(n,) or ports['output_index'].ndim!=2 or ports['output_index'].shape!=ports['output_scale'].shape:
            raise ValueError('Port shapes do not match the circuit')
        if any(ports[k].dtype.kind not in 'iu' for k in ['input_channel','output_index']) or (ports['input_channel']<-1).any() or (ports['output_index']<0).any() or (ports['output_index']>=n).any():
            raise ValueError('Invalid neuron or input-channel indices')
        if not all(np.isfinite(ports[k]).all() for k in ['input_scale','output_scale']):
            raise ValueError('Port scales must be finite')
        ptr, idx = w.crow_indices().int(), w.col_indices().int()
        rows = torch.repeat_interleave(torch.arange(n, dtype=torch.int32), ptr.diff().long())
        order = torch.argsort(idx, stable=True).int()
        tptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.bincount(idx.long(), minlength=n).cumsum(0).int()])
        for name, value in dict(ptr=ptr, idx=idx, rows=rows, tptr=tptr,
                                tidx=rows[order.long()], order=order, base=w.values()).items():
            self.register_buffer(name, value, persistent=False)
        for name, value in ports.items():
            self.register_buffer(name, torch.from_numpy(np.asarray(value)), persistent=False)
        self.calibrated = calibration is not None
        if self.calibrated and control!='real':raise ValueError('Prepare matching cell-type groups before using a rewired control')
        if self.calibrated:
            with np.load(calibration, allow_pickle=False) as f:
                edge, neuron = f['edge_group'], f['neuron_group']
            for name, array, length in [('edge_group', edge, len(idx)), ('neuron_group', neuron, n)]:
                if array.shape != (length,) or array.dtype.kind not in 'iu' or array.min() < 0 or array.max() >= length:
                    raise ValueError('Invalid anatomical parameter groups')
                self.register_buffer(name, torch.from_numpy(array.astype(np.int64)), persistent=False)
        self.edge_gain = nn.Parameter(torch.zeros(int(edge.max())+1 if self.calibrated else len(idx)))
        self.bias = nn.Parameter(torch.zeros(int(neuron.max())+1 if self.calibrated else n))
        self.leak = nn.Parameter(torch.zeros(int(neuron.max())+1 if self.calibrated else n))
        self.n_neurons, self.n_edges = n, len(idx)

    def initial(self, batch):
        return self.bias.new_zeros((self.n_neurons, batch))

    def coefficients(self):
        gain = self.edge_gain[self.edge_group] if self.calibrated else self.edge_gain
        values=self.base*(gain.clamp(-3, 3).exp() if self.calibrated else 2*gain.sigmoid())
        leak=self.leak[self.neuron_group] if self.calibrated else self.leak
        leak=.05+.9*leak.sigmoid()[:,None]
        transposed=values.new_empty(0)
        if torch.is_grad_enabled():
            with torch.no_grad():transposed=values[self.order.long()]
        return values,leak,transposed

    def forward(self, signals, state=None, ticks=8, ablate=False, coefficients=None):
        if state is None:
            state = self.initial(len(signals))
        if signals.ndim != 2 or state.shape != (self.n_neurons, len(signals)):
            raise ValueError("Circuit expects batched input channels and matching neural state")
        # A neuron receives at most one external channel; unmapped neurons get zero.
        drive = signals[:, self.input_channel.clamp_min(0).long()].t() * self.input_scale[:, None]
        values,leak,transposed=self.coefficients() if coefficients is None else coefficients
        bias = self.bias[self.neuron_group] if self.calibrated else self.bias
        for _ in range(ticks):
            recurrent = 0 if ablate else EdgeMultiply.apply(state, values, self.ptr, self.idx, self.rows, self.tptr, self.tidx, transposed)
            state = state + leak * (torch.tanh(drive + bias[:, None] + recurrent) - state)
        return state

    def read(self, state):
        # Outputs pool declared neurons only. Signs/calibration belong to fixed ports.
        return (state[self.output_index.long()] * self.output_scale[..., None]).sum(1).t()

    def regularization(self):
        return self.edge_gain.square().mean() + .1 * (self.bias.square().mean() + self.leak.square().mean())
