"""Replay fixed-shape inference without launching each layer from Python."""
import torch
import torch.nn.functional as F


class CudaEvaluator:
    def __init__(self, model, device):
        self.model, self.device = model, device
        self.graphs = {}

    def __call__(self, states, legal):
        size = len(states)
        with torch.no_grad():
            if size not in self.graphs:
                inputs = torch.zeros((size, 3, 6, 7), device=self.device)
                masks = torch.ones((size, 7), dtype=torch.bool, device=self.device)

                def forward():
                    logits, values = self.model(inputs)
                    policy = F.softmax(logits.masked_fill(~masks, float('-inf')), dim=-1)
                    return torch.cat((policy, values.reshape(size, 1)), dim=1)

                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        forward()
                torch.cuda.current_stream(self.device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = forward()
                self.graphs[size] = (graph, inputs, masks, output)
            graph, inputs, masks, output = self.graphs[size]
            inputs.copy_(torch.from_numpy(states))
            masks.copy_(torch.from_numpy(legal))
            graph.replay()
            result = output.cpu().numpy()
            # The native ABI expects packed arrays, rather than column views of an eight-column buffer.
            return result[:, :7].copy(), result[:, 7].copy()
