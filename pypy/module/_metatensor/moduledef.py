from pypy.interpreter.mixedmodule import MixedModule


class Module(MixedModule):

    interpleveldefs = {
        'Tensor': 'interp_tensor.W_Tensor',
        '_tensor_flat': 'interp_tensor.tensor_flat',
        'zeros': 'interp_tensor.zeros',
        'cat': 'interp_tensor.cat',
        'basic_index': 'interp_tensor.basic_index',
        'box_of': 'interp_tensor.box_of',
        'same_shape': 'interp_tensor.same_shape',
        'numel': 'interp_tensor.numel',
        'broadcast_shape': 'interp_tensor.broadcast_shape',
        'contiguous_strides': 'interp_tensor.contiguous_strides',
        'spread_strides': 'interp_tensor.spread_strides',
        'expand_layout': 'interp_tensor.expand_layout',
        'infer_shape': 'interp_tensor.infer_shape',
        'block_span': 'interp_tensor.block_span',
        'permute_layout': 'interp_tensor.permute_layout',
        'scalar': 'interp_tensor.scalar',
        'kernel_count': 'interp_tensor.kernel_count',
        'kernel_compile_count': 'interp_tensor.kernel_compile_count',
        'launch_count': 'interp_tensor.launch_count',
        'lazy_enabled': 'interp_tensor.lazy_enabled',
        'lazy_stats': 'interp_tensor.lazy_stats',
        'mark_step': 'interp_tensor.mark_step',
        'mem_total': 'interp_tensor.mem_total',
        'live_bytes': 'interp_tensor.live_bytes',
        'alloc_failed': 'interp_tensor.alloc_failed',
        'cpu_fallbacks': 'interp_tensor.cpu_fallbacks',
        'profile_report': 'interp_tensor.profile_report',
        'unfused_fallbacks': 'interp_tensor.unfused_fallbacks',
        'group': 'interp_group.new_group',  # [motion]
    }

    appleveldefs = {
        'tensor': 'app_tensor.tensor',
    }
