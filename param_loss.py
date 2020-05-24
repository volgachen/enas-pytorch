import torch
import torch.nn

class ParamCalculation(object):
    def __init__(self, target_param, num_layers=9, num_cells=5, out_filters=36):
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.out_filters = out_filters  
        pool_distance = self.num_layers // 3
        self.pool_layers = [pool_distance, 2*pool_distance+1]

        self.target = target_param

    def getLoss(self, arch):
        param = self.checkparam(arch)
        param = param * 4.0 / 1024 / 1024
        loss =  -(param - self.target)**2/2.0
        return loss

    def checkparam(self, arch):
        out_filters = self.out_filters
        stem_param = 3 * (out_filters * 3) * 3 * 3 + out_filters * 4

        normal_arc, reduce_arc = arch
        in_filters = [out_filters*3, out_filters*3]
        sizes = [1, 1]
        layer_params = []
        for layer_id in range(self.num_layers+2):
            param_this_layer = 0
            if layer_id not in self.pool_layers:
                param_this_layer += self._calc_layer(layer_id, in_filters, out_filters, sizes, normal_arc)
            else:
                out_filters *= 2
                param_this_layer += self._calc_reduction(in_filters[-1], out_filters)
                in_filters = [in_filters[-1], out_filters]
                sizes = [sizes[-1], sizes[-1] * 2]
                param_this_layer += self._calc_layer(layer_id, in_filters, out_filters, sizes, reduce_arc)
                sizes = [sizes[-1], sizes[-1] * 2]
            layer_params.append(param_this_layer)
            in_filters = [in_filters[-1], out_filters]
        return stem_param + sum(layer_params)

    def _calc_reduction(self, in_filters, out_filters):
        return 1 * 1 * in_filters * out_filters + 4 * out_filters

    def _calc_layer(self, layer_id, in_filters, out_filters, sizes, arc):
        params = self._calc_calibrate(in_filters, out_filters, sizes)
        used = torch.zeros(self.num_cells+2).long()
        for cell_id in range(self.num_cells):
            x_id, x_op, y_id, y_op = arc[4 * cell_id: 4* cell_id + 4]
            used[x_id] = 1
            used[y_id] = 1
            if x_op < 2:
                params += self._calc_conv(3 if x_op==0 else 5, out_filters)
            if y_op < 2:
                params += self._calc_conv(3 if y_op==0 else 5, out_filters)
        indices = torch.eq(used, 0).nonzero().long().view(-1)
        num_outs = indices.size(0)
        # final conv and bn
        params += 1 * 1 * out_filters * (out_filters * num_outs)
        params += 4 * out_filters
        return params

    def _calc_calibrate(self, in_filters, out_filters, sizes):
        params = 0
        if sizes[0] * 2 == sizes[1]:
            params +=self._calc_reduction(in_filters[0], out_filters)
        elif in_filters[0] != out_filters:
            params += 1 * 1 * in_filters[0] * out_filters + 4 * out_filters
        if in_filters[1] != out_filters:
            params += 1 * 1 * in_filters[1] * out_filters + 4 * out_filters
        return params

    def _calc_conv(self, filter_size, out_filter):
        params = 0
        params += filter_size * filter_size * out_filter
        params += 1 * 1 * out_filter * out_filter
        params += 4 * out_filter
        params *= 2
        return params