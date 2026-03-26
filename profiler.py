import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from megatron.core import mpu
from mindspeed.core.context_parallel import model_parallel_utils as ms_mpu
from megatron.training.training import setup_model_and_optimizer
from mindspeed_mm.training import no_wd_decay_cond, scale_lr_cond
from megatron.core.enums import ModelType

@dataclass
class ModuleProfileInfo:
    """模块性能分析信息"""
    name: str
    module_path: str
    module_type: str
    forward_times: List[float] = None
    backward_times: List[float] = None
    seq_lengths: List[int] = None
    forward_coeffs: Tuple[float, float, float] = None  # (a, b, c) for ax^2 + bx + c
    backward_coeffs: Tuple[float, float, float] = None
    
    def __post_init__(self):
        if self.forward_times is None:
            self.forward_times = []
        if self.backward_times is None:
            self.backward_times = []
        if self.seq_lengths is None:
            self.seq_lengths = []

MODEL_MODULE_INFO = {
    "InternVL": {
        "vision_embedding": {
            "module_paths": ["module.module.image_encoder.encoder.embeddings"],
            "module_type": "embedding",
            "layer_num": 1
        },
        "vision_transformer_layer": {
            "module_paths": [f"module.module.image_encoder.encoder.encoder.layers.{i}" for i in range(4)],
            "module_type": "transformer_layer",
            "layer_num": 1
        },
        "vl_mlp": {
            "module_paths": ["module.module.image_encoder.projector"],
            "module_type": "mlp",
            "layer_num": 1
        },
        "text_embedding": {
            "module_paths": ["module.module.text_decoder.embedding"],
            "module_type": "embedding",
            "layer_num": 1
        },
        "text_transformer_layer": {
            "module_paths": [f"module.module.text_decoder.decoder.layers.{i}" for i in range(12)],
            "module_type": "transformer_layer",
            "layer_num": 1
        },
        "text_output_layer": {
            "module_paths": ["module.module.text_decoder.output_layer"],
            "module_type": "output_layer",
            "layer_num": 1
        }
    }
    # 可以在这里添加其他模型的模块信息
}

class MLLMProfiler:
    
    def __init__(
        self, 
        model_name, 
        model, 
        args,
        cluster_size: int = 8, 
        img_context_token_id = 151667,
        device: str = "npu"
    ):
        """
        初始化性能分析器
        
        Args:
            model_name: 模型名称，对应MODEL_MODULE_INFO中的键
            device: 运行设备
        """

        self.model_name = model_name
        lr_mult = args.lr_mult
        
        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            lambda pre_process=True, post_process=True, modules=None: model, ModelType.encoder_or_decoder, no_wd_decay_cond=no_wd_decay_cond, scale_lr_cond=scale_lr_cond, lr_mult=lr_mult)
        self.model = model[0]

        self.cluster_size = cluster_size
        self.device=device
        
        if model_name not in MODEL_MODULE_INFO:
            raise ValueError(f"Model {model_name} is not defined in MODEL_MODULE_INFO。")
        # torch.set_default_dtype(torch.bfloat16)

        self.module_info = MODEL_MODULE_INFO[model_name]
        self.profile_data_tmp: Dict[str, ModuleProfileInfo] = {}
        self.profile_data: Dict[str, ModuleProfileInfo] = {}
        self.hooks = []
        self.is_profil = False
        self.img_context_token_id = img_context_token_id
        self._init_module_profile_info()
        self.profile([8192])
        self.fit()
        torch.distributed.breakpoint()

    def _init_module_profile_info(self):
        """初始化模块性能分析信息"""
        for module_type, info in self.module_info.items():
            for i, path in enumerate(info["module_paths"]):
                module_name = f"{module_type}_{i}" if len(info["module_paths"]) > 1 else module_type
                self.profile_data_tmp[module_name] = ModuleProfileInfo(
                    name=module_name,
                    module_path=path,
                    module_type=info["module_type"],
                )
                self.profile_data[module_name] = ModuleProfileInfo(
                    name=module_name,
                    module_path=path,
                    module_type=info["module_type"],
                )

    def _get_module_by_path(self, model: nn.Module, path: str) -> nn.Module:
        """根据路径获取模块"""
        modules = path.split(".")
        current_module = model
        
        for module_name in modules:
            if hasattr(current_module, module_name):
                current_module = getattr(current_module, module_name)
            elif hasattr(current_module, "_modules") and module_name in current_module._modules:
                current_module = current_module._modules[module_name]
            else:
                try:
                    idx = int(module_name)
                    if isinstance(current_module, (nn.ModuleList, nn.Sequential)):
                        current_module = current_module[idx]
                    else:
                        raise AttributeError(f"无法通过索引访问模块: {module_name}")
                except ValueError:
                    raise AttributeError(f"未找到模块: {module_name} 在路径 {path}")
        
        return current_module

    def _register_hooks(self, model: nn.Module):
        """为指定模块注册前向和反向钩子"""
        
        def create_forward_hook(module_name: str):
            def forward_hook(module, input, output):
                if not hasattr(module, '_forward_start_time'):
                    return
                forward_time = time.time() - module._forward_start_time
                torch.npu.synchronize()
                self.profile_data_tmp[module_name].forward_times.append(forward_time)
            return forward_hook
        
        def create_backward_hook(module_name: str):
            def backward_hook(module, grad_input, grad_output):
                if not hasattr(module, '_backward_start_time'):
                    return
                torch.npu.synchronize()
                backward_time = time.time() - module._backward_start_time
                self.profile_data_tmp[module_name].backward_times.append(backward_time)
            return backward_hook
        
        def create_pre_forward_hook(module_name: str):
            def pre_forward_hook(module, input):
                module._forward_start_time = time.time()
            return pre_forward_hook
        
        def create_pre_backward_hook(module_name: str):
            def pre_backward_hook(module, grad_output):
                module._backward_start_time = time.time()
            return pre_backward_hook
        
        for module_name, info in self.profile_data_tmp.items():
            try:
                module = self._get_module_by_path(model, info.module_path)
                
                pre_forward_hook = module.register_forward_pre_hook(create_pre_forward_hook(module_name))
                forward_hook = module.register_forward_hook(create_forward_hook(module_name))
                
                pre_backward_hook = module.register_full_backward_pre_hook(create_pre_backward_hook(module_name))
                backward_hook = module.register_full_backward_hook(create_backward_hook(module_name))
                
                self.hooks.extend([pre_forward_hook, forward_hook, pre_backward_hook, backward_hook])
                
            except AttributeError as e:
                print(f"警告: 无法为模块 {info.module_path} 注册钩子: {e}")

    def _remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def profile(self, seq_lengths: List[int] = None, 
                num_warmup: int = 2, num_repeat: int = 3):
        """
        对模型进行性能分析
        
        Args:
            model: 要分析的模型
            seq_lengths: 要分析的序列长度列表，默认为[64, 128, 256, 512, 1024]
            num_warmup: 预热次数
            num_repeat: 每个序列长度重复次数
        """
        if seq_lengths is None:
            seq_lengths = [64, 128, 256, 512, 1024]
        
        # model = self.model.to(self.device)
        model = self.model
        model.train()  # 设置为训练模式以启用反向传播
        

        self._register_hooks(model)
        print("开始预热...")
        for _ in range(num_warmup):
            inputs = self._create_inputs(seq_lengths[0])
            outputs = model(**inputs)
            loss = outputs['loss_dict']['loss']
            loss.backward()

        torch.distributed.barrier()
        
        for info in self.profile_data_tmp.values():
            info.forward_times.clear()
            info.backward_times.clear()
            info.seq_lengths.clear()
        
        if (torch.distributed.get_rank() == 0): print("开始性能分析...")
        
        for seq_len in seq_lengths:
            for cp_size in range(seq_len // 4096, seq_len // 2048 + 1):
                if (torch.distributed.get_rank() == 0): print(f"分析序列长度: {seq_len}, CP_SIZE: {cp_size}")
                local_group_ranks = [r for r in range(cp_size)]
                group = mpu.create_group(ranks=local_group_ranks, use_local_synchronization=False)
                # torch.distributed.breakpoint()
                if (torch.distributed.get_rank() < cp_size):
                    mpu._CONTEXT_PARALLEL_GROUP = group
                    mpu._CONTEXT_PARALLEL_GLOBAL_RANKS = local_group_ranks
                    torch.distributed.barrier(group)
                    ms_mpu._CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_KV = local_group_ranks
                    ms_mpu._CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_DKV = local_group_ranks
                    for repeat in range(num_repeat):
                        print(f"Repeat Num {repeat}")
                        cp_size_2 = cp_size * 2
                        inputs = self._create_inputs(((seq_len + cp_size_2 - 1) // cp_size_2) * cp_size_2)

                        model.zero_grad()
                         
                        torch.npu.synchronize() if self.device == "npu" else None
                        start_time = time.time()
                        
                        outputs = model(**inputs)
                        loss = outputs['loss_dict']['loss']
                        
                        torch.npu.synchronize() if self.device == "npu" else None
                        forward_time = time.time() - start_time
                        print("Forward finished")
                        
                        torch.npu.synchronize() if self.device == "npu" else None
                        start_time = time.time()
                        
                        loss.backward()
                        
                        torch.npu.synchronize() if self.device == "npu" else None
                        backward_time = time.time() - start_time
                        
                    for name, info in self.profile_data_tmp.items():
                        info.forward_times = [forward_time * cp_size for forward_time in info.forward_times]
                        info.backward_times = [backward_time * cp_size for backward_time in info.backward_times]

                    for name, info in self.profile_data.items():
                        info.forward_times.append(
                            sum(self.profile_data_tmp[name].forward_times) / len(self.profile_data_tmp[name].forward_times)
                        )
                        info.backward_times.append(
                            sum(self.profile_data_tmp[name].backward_times) / len(self.profile_data_tmp[name].backward_times)
                        )
                        if len(info.seq_lengths) < repeat + 1:
                            info.seq_lengths.append(seq_len)
                torch.distributed.barrier()
                torch.distributed.destroy_process_group(group)
        
        self._remove_hooks()
        
        self.is_profiled = True
        print("性能分析完成!")

    def _create_inputs(self, seq_length: int, batch_size: int = 1, 
                      image_patch_size: Tuple[int, int] = (448, 448), vision_tokens_ratio: float = 0.8) -> Dict:
        """
        创建随机输入数据
        
        Args:
            seq_length: 序列长度
            batch_size: 批大小
            image_patch_size: 分块图像大小，请根据模型架构自行设定
            
        Returns:
            输入字典
        """
        # 创建图像输入
        vision_tokens_per_patch = 256
        num_patches = (int(seq_length * vision_tokens_ratio) // vision_tokens_per_patch)
        channel = 3
        pixel_values = torch.randn((batch_size * num_patches, 3, *image_patch_size), device=self.device, dtype=torch.bfloat16)
        
        # 创建文本输入 (假设词汇表大小为151674，根据InternVL的output_layer维度)
        input_ids = torch.randint(0, 151650, (batch_size, seq_length), device=self.device)
        input_ids[:, -num_patches * vision_tokens_per_patch:] = self.img_context_token_id
        
        # 创建注意力掩码
        attention_mask = torch.ones(batch_size, seq_length, device=self.device, dtype=torch.bool)
        
        # 创建标签 (用于计算损失)
        labels = torch.randint(0, 151674, (batch_size, seq_length), device=self.device)
        
        # 创建图像标志 (假设为二元标志)
        image_flags = torch.ones([batch_size * num_patches], dtype=torch.long, device=self.device)
        
        return {
            'pixel_values': pixel_values,
            'image_flags': image_flags,
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'transfer': None
        }

    def fit(self):
        """
        根据收集的数据拟合二次函数
        """
        if not self.is_profiled:
            raise RuntimeError("请先调用profile方法收集数据")
        
        print("开始拟合二次函数...")
        
        for module_name, info in self.profile_data.items():
            if len(info.seq_lengths) > 0 and len(info.forward_times) > 0:
                info.forward_coeffs = self._fit_quadratic(
                    info.seq_lengths, 
                    info.forward_times
                )
                torch.distributed.breakpoint()
                if len(info.backward_times) > 0:
                    info.backward_coeffs = self._fit_quadratic(
                        info.seq_lengths,
                        info.backward_times
                    )
                
                print(f"模块 {module_name}:")
                print(f"  前向传播: {info.forward_coeffs}")
                print(f"  反向传播: {info.backward_coeffs}")

    def _fit_quadratic(self, x: List[float], y: List[float]) -> Tuple[float, float, float]:
        """
        用二次函数拟合数据: y = ax^2 + bx + c
        
        Args:
            x: 输入序列
            y: 输出序列
            
        Returns:
            二次函数系数 (a, b, c)
        """
        if len(x) < 3:
            if len(x) == 0:
                return (0.0, 0.0, 0.0)
            coeffs = np.polyfit(x, y, min(1, len(x)-1))
            coeffs = list(coeffs) + [0.0] * (3 - len(coeffs))
            return tuple(coeffs[::-1])  

        coeffs = np.polyfit(x, y, 2)
        return tuple(coeffs[::-1]) 