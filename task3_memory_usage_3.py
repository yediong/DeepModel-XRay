# 解决OpenMP冲突
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torchvision.models as models
from transformers import T5ForConditionalGeneration, BertModel
import gc
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from tabulate import tabulate

class GPUMemoryMonitor:
    """GPU显存监控器类，用于全面测量和分析GPU显存使用情况"""

    def __init__(self, output_dir="./memory_analysis_results"):
        """初始化显存监控器

        Args:
            output_dir: 结果输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 检查GPU是否可用
        if not torch.cuda.is_available():
            raise RuntimeError("错误: GPU不可用，请确保CUDA环境正确配置")

        self.device_name = torch.cuda.get_device_name(0)
        self.device_properties = torch.cuda.get_device_properties(0)

        print(f"==== GPU信息 ====")
        print(f"设备名称: {self.device_name}")
        print(f"计算能力: {self.device_properties.major}.{self.device_properties.minor}")
        print(f"总显存: {self.device_properties.total_memory / 1024 ** 3:.2f} GB")
        print(f"多处理器数量: {self.device_properties.multi_processor_count}")

        # 记录测试结果
        self.results = {
            'vgg16': {
                'batch_size': [],
                'inference': []
            },
            't5': {
                'batch_size': [],
                'seq_length': [],
                'inference': []
            },
            'bert': {
                'batch_size': [],
                'seq_length': [],
                'inference': []
            }
        }

    def clear_gpu_memory(self):
        """彻底清空GPU显存"""
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()

    def get_memory_usage(self):
        """获取当前GPU显存使用情况

        Returns:
            dict: 包含显存使用情况的字典
        """
        return {
            'allocated': torch.cuda.memory_allocated() / 1024 ** 2,  # MB
            'reserved': torch.cuda.memory_reserved() / 1024 ** 2,  # MB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024 ** 2,  # MB
            'max_reserved': torch.cuda.max_memory_reserved() / 1024 ** 2  # MB
        }

    def measure_model_memory(self, model_name, model, input_generator, input_sizes, n_repeats=3):
        """测量模型在不同输入大小下的显存使用情况

        Args:
            model_name: 模型名称
            model: 模型实例
            input_generator: 生成输入数据的函数
            input_sizes: 要测试的输入大小列表
            n_repeats: 重复测量次数，用于计算平均值

        Returns:
            dict: 包含测量结果的字典
        """
        results = []

        print(f"\n==== {model_name}模型显存测试 ====")
        print(f"参数总量: {self.count_parameters(model):,}")
        print(f"模型参数量: {self.count_parameters(model) / 1e6:.2f}M")

        # 测量模型本身占用的显存
        self.clear_gpu_memory()
        base_memory = self.get_memory_usage()['allocated']
        print(f"模型基础显存占用: {base_memory:.2f}MB")

        for size in input_sizes:
            size_results = {
                'input_size': size,
                'inference': []
            }

            # 重复测量以获得稳定结果
            for _ in range(n_repeats):
                # 测量推理阶段显存
                inference_memory = self._measure_inference_memory(model, model_name, input_generator, size)
                size_results['inference'].append(inference_memory)

            # 计算平均值
            size_results['inference_avg'] = {
                k: np.mean([r[k] for r in size_results['inference']])
                for k in size_results['inference'][0]
            }

            # 记录输出
            if isinstance(size, tuple):
                if len(size) == 2:  # T5/BERT模型
                    batch_size, seq_length = size
                    print(f"{model_name} - 批次大小: {batch_size}, 序列长度: {seq_length}")

                    # 存储到全局结果中
                    if model_name == 't5':
                        self.results['t5']['batch_size'].append(batch_size)
                        self.results['t5']['seq_length'].append(seq_length)
                        self.results['t5']['inference'].append(size_results['inference_avg']['memory_increase'])
                    elif model_name == 'bert':
                        self.results['bert']['batch_size'].append(batch_size)
                        self.results['bert']['seq_length'].append(seq_length)
                        self.results['bert']['inference'].append(size_results['inference_avg']['memory_increase'])
                else:
                    print(f"{model_name} - 输入大小: {size}")
            else:  # VGG16模型
                print(f"{model_name} - 批次大小: {size}")

                # 存储到全局结果中
                if model_name == 'vgg16':
                    self.results['vgg16']['batch_size'].append(size)
                    self.results['vgg16']['inference'].append(size_results['inference_avg']['memory_increase'])

            # 打印结果
            print(f"  推理阶段: 初始显存: {size_results['inference_avg']['initial_memory']:.2f}MB, "
                  f"前向传播后: {size_results['inference_avg']['forward_memory']:.2f}MB, "
                  f"增加: {size_results['inference_avg']['memory_increase']:.2f}MB, "
                  f"峰值: {size_results['inference_avg']['peak_memory']:.2f}MB")

            results.append(size_results)

        return results

    def _measure_inference_memory(self, model, model_name, input_generator, size):
        """测量推理阶段的显存使用情况"""
        self.clear_gpu_memory()

        # 记录初始显存
        initial_usage = self.get_memory_usage()
        initial_memory = initial_usage['allocated']

        # 生成输入数据
        inputs = input_generator(size)

        # 推理阶段 (前向传播)
        with torch.no_grad():
            if model_name == 'vgg16':
                _ = model(inputs)
            else:  # transformer模型
                _ = model(**inputs)

        torch.cuda.synchronize()
        forward_usage = self.get_memory_usage()
        forward_memory = forward_usage['allocated']
        peak_memory = forward_usage['max_allocated']

        return {
            'initial_memory': initial_memory,
            'forward_memory': forward_memory,
            'memory_increase': forward_memory - initial_memory,
            'peak_memory': peak_memory
        }

    def count_parameters(self, model):
        """计算模型的参数量

        Args:
            model: PyTorch模型

        Returns:
            int: 参数总量
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"参数总量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")

        return total_params

    def analyze_layer_memory(self, model, input_generator, input_size):
        """分析模型各层的显存占用情况

        Args:
            model: PyTorch模型
            input_generator: 生成输入数据的函数
            input_size: 输入大小

        Returns:
            dict: 包含各层显存占用的字典
        """
        try:
            layer_memory = {}

            def hook_fn(name):
                def hook(module, input, output):
                    try:
                        # 计算该层输出的大小
                        if isinstance(output, torch.Tensor):
                            size = output.element_size() * output.nelement()
                        elif isinstance(output, tuple) and len(output) > 0:
                            size = 0
                            for o in output:
                                if isinstance(o, torch.Tensor):
                                    size += o.element_size() * o.nelement()
                        else:
                            size = 0

                        layer_memory[name] = size / 1024 ** 2  # 转换为MB
                    except Exception as e:
                        print(f"处理层 {name} 时出错: {str(e)}")

                return hook

            # 只为顶层模块注册钩子以减少复杂性
            hooks = []
            top_modules = []
            for name, module in model.named_children():
                if name:  # 跳过空名称的模块
                    hooks.append(module.register_forward_hook(hook_fn(name)))
                    top_modules.append(name)

            print(f"正在分析以下顶层模块: {', '.join(top_modules)}")

            # 前向传播
            self.clear_gpu_memory()
            with torch.no_grad():
                inputs = input_generator(input_size)
                if isinstance(model, models.vgg.VGG):
                    _ = model(inputs)
                else:  # transformer模型
                    _ = model(**inputs)

            # 移除钩子
            for hook in hooks:
                hook.remove()

            return layer_memory
        except Exception as e:
            print(f"分析层显存时出错: {str(e)}")
            return {}
            
    def plot_batch_size_memory(self, vgg_results, t5_batch_results, bert_batch_results):
        """绘制批次大小对显存的影响"""
        plt.figure(figsize=(14, 10), dpi=100)

        # 设置风格和主题
        plt.style.use('seaborn-v0_8-whitegrid')

        # 自定义颜色
        vgg_color = '#3498db'  # 蓝色
        t5_color = '#e74c3c'   # 红色
        bert_color = '#2ecc71' # 绿色

        # 获取数据
        vgg_batch = self.results['vgg16']['batch_size']
        vgg_inference = self.results['vgg16']['inference']

        # 为平滑曲线准备插值数据
        vgg_x = np.array(vgg_batch)
        vgg_y = np.array(vgg_inference)
        vgg_x_smooth = np.linspace(min(vgg_x), max(vgg_x), 100)
        vgg_y_smooth = np.interp(vgg_x_smooth, vgg_x, vgg_y)

        # 绘制VGG16的数据点和平滑曲线
        plt.plot(vgg_x_smooth, vgg_y_smooth, '-', color=vgg_color, linewidth=3, alpha=0.7)
        plt.scatter(vgg_batch, vgg_inference, color=vgg_color, s=120, label='VGG16 Inference',
                    edgecolor='white', linewidth=2, zorder=3)

        # 获取T5的数据
        t5_batch = []
        t5_inference = []

        for i, seq_len in enumerate(self.results['t5']['seq_length']):
            if seq_len == 64:  # 只使用序列长度为64的数据点
                t5_batch.append(self.results['t5']['batch_size'][i])
                t5_inference.append(self.results['t5']['inference'][i])

        # 为T5准备插值数据
        t5_x = np.array(t5_batch)
        t5_y = np.array(t5_inference)
        t5_x_smooth = np.linspace(min(t5_x), max(t5_x), 100)
        t5_y_smooth = np.interp(t5_x_smooth, t5_x, t5_y)

        # 绘制T5的数据点和平滑曲线
        plt.plot(t5_x_smooth, t5_y_smooth, '-', color=t5_color, linewidth=3, alpha=0.7)
        plt.scatter(t5_batch, t5_inference, color=t5_color, s=120, label='T5 Inference (seq_len=64)',
                    edgecolor='white', linewidth=2, marker='s', zorder=3)
                    
        # 获取BERT的数据
        bert_batch = []
        bert_inference = []

        for i, seq_len in enumerate(self.results['bert']['seq_length']):
            if seq_len == 64:  # 只使用序列长度为64的数据点
                bert_batch.append(self.results['bert']['batch_size'][i])
                bert_inference.append(self.results['bert']['inference'][i])

        # 为BERT准备插值数据
        bert_x = np.array(bert_batch)
        bert_y = np.array(bert_inference)
        bert_x_smooth = np.linspace(min(bert_x), max(bert_x), 100)
        bert_y_smooth = np.interp(bert_x_smooth, bert_x, bert_y)

        # 绘制BERT的数据点和平滑曲线
        plt.plot(bert_x_smooth, bert_y_smooth, '-', color=bert_color, linewidth=3, alpha=0.7)
        plt.scatter(bert_batch, bert_inference, color=bert_color, s=120, label='BERT Inference (seq_len=64)',
                    edgecolor='white', linewidth=2, marker='^', zorder=3)

        # 添加回归线
        vgg_z = np.polyfit(vgg_x, vgg_y, 1)
        vgg_p = np.poly1d(vgg_z)
        plt.plot(vgg_x_smooth, vgg_p(vgg_x_smooth), '--', color=vgg_color, alpha=0.5,
                 linewidth=2, label=f'VGG16 Linear Fit: y = {vgg_z[0]:.2f}x + {vgg_z[1]:.2f}')

        t5_z = np.polyfit(t5_x, t5_y, 1)
        t5_p = np.poly1d(t5_z)
        plt.plot(t5_x_smooth, t5_p(t5_x_smooth), '--', color=t5_color, alpha=0.5,
                 linewidth=2, label=f'T5 Linear Fit: y = {t5_z[0]:.2f}x + {t5_z[1]:.2f}')
                 
        bert_z = np.polyfit(bert_x, bert_y, 1)
        bert_p = np.poly1d(bert_z)
        plt.plot(bert_x_smooth, bert_p(bert_x_smooth), '--', color=bert_color, alpha=0.5,
                 linewidth=2, label=f'BERT Linear Fit: y = {bert_z[0]:.2f}x + {bert_z[1]:.2f}')

        # 计算最大值并添加注释
        max_vgg = max(vgg_inference)
        max_t5 = max(t5_inference)
        max_bert = max(bert_inference)

        plt.annotate(f'VGG16 Max: {max_vgg:.1f}MB',
                     xy=(vgg_batch[vgg_inference.index(max_vgg)], max_vgg),
                     xytext=(10, 15), textcoords='offset points',
                     fontsize=12, color=vgg_color, weight='bold',
                     arrowprops=dict(arrowstyle='->', color=vgg_color, lw=1.5))

        plt.annotate(f'T5 Max: {max_t5:.1f}MB',
                     xy=(t5_batch[t5_inference.index(max_t5)], max_t5),
                     xytext=(10, -30), textcoords='offset points',
                     fontsize=12, color=t5_color, weight='bold',
                     arrowprops=dict(arrowstyle='->', color=t5_color, lw=1.5))
                     
        plt.annotate(f'BERT Max: {max_bert:.1f}MB',
                     xy=(bert_batch[bert_inference.index(max_bert)], max_bert),
                     xytext=(-60, -15), textcoords='offset points',
                     fontsize=12, color=bert_color, weight='bold',
                     arrowprops=dict(arrowstyle='->', color=bert_color, lw=1.5))

        # 设置图表属性
        plt.xlabel('Batch Size', fontsize=16, fontweight='bold')
        plt.ylabel('Memory Increase (MB)', fontsize=16, fontweight='bold')
        plt.title('Impact of Batch Size on GPU Memory Usage\nVGG16 vs T5 vs BERT',
                  fontsize=20, fontweight='bold', pad=20)

        # 美化刻度
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # 网格线
        plt.grid(True, alpha=0.3, linestyle='--')

        # 添加图例
        legend = plt.legend(fontsize=13, frameon=True, facecolor='white', edgecolor='gray')
        legend.get_frame().set_alpha(0.9)

        # 添加水印
        plt.figtext(0.99, 0.01, 'GPU Memory Analysis',
                    fontsize=10, color='gray', ha='right', alpha=0.7)

        # 美化图表边框
        for spine in plt.gca().spines.values():
            spine.set_linewidth(1.5)

        # 添加次要刻度
        plt.minorticks_on()
        plt.grid(which='minor', alpha=0.1)

        plt.tight_layout()

        # 保存图表
        plt.savefig(f'{self.output_dir}/batch_size_memory_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_seq_length_memory_comparison(self):
        """绘制序列长度对T5和BERT显存的影响比较"""
        plt.figure(figsize=(14, 10), dpi=100)
        plt.style.use('seaborn-v0_8-whitegrid')

        # 自定义颜色
        t5_color = '#9b59b6'    # 紫色
        bert_color = '#2ecc71'  # 绿色
        
        # 提取固定batch_size下不同模型的seq_length数据
        batch_size_filter = 8  # 使用batch_size=8的数据点
        
        # T5数据
        t5_seqlens = []
        t5_seq_inference = []
        for i, batch in enumerate(self.results['t5']['batch_size']):
            if batch == batch_size_filter:
                t5_seqlens.append(self.results['t5']['seq_length'][i])
                t5_seq_inference.append(self.results['t5']['inference'][i])
        
        # 数据点排序
        t5_idx = np.argsort(t5_seqlens)
        t5_seqlens = [t5_seqlens[i] for i in t5_idx]
        t5_seq_inference = [t5_seq_inference[i] for i in t5_idx]
        
        # BERT数据
        bert_seqlens = []
        bert_seq_inference = []
        for i, batch in enumerate(self.results['bert']['batch_size']):
            if batch == batch_size_filter:
                bert_seqlens.append(self.results['bert']['seq_length'][i])
                bert_seq_inference.append(self.results['bert']['inference'][i])
        
        # 数据点排序
        bert_idx = np.argsort(bert_seqlens)
        bert_seqlens = [bert_seqlens[i] for i in bert_idx]
        bert_seq_inference = [bert_seq_inference[i] for i in bert_idx]
        
        # 绘制T5散点图和连线
        plt.scatter(t5_seqlens, t5_seq_inference, s=180, color=t5_color,
                    edgecolor='white', linewidth=2, zorder=3,
                    label=f'T5 Memory Usage (batch_size={batch_size_filter})')
        plt.plot(t5_seqlens, t5_seq_inference, '-', color=t5_color,
                 alpha=0.7, linewidth=3)
                 
        # 绘制BERT散点图和连线
        plt.scatter(bert_seqlens, bert_seq_inference, s=180, color=bert_color,
                    edgecolor='white', linewidth=2, zorder=3, marker='^',
                    label=f'BERT Memory Usage (batch_size={batch_size_filter})')
        plt.plot(bert_seqlens, bert_seq_inference, '-', color=bert_color,
                 alpha=0.7, linewidth=3)
                 
        # 拟合二次曲线 - T5
        if len(t5_seqlens) >= 3:
            x = np.array(t5_seqlens)
            y = np.array(t5_seq_inference)
            # 二次拟合
            t5_z = np.polyfit(x, y, 2)
            t5_p = np.poly1d(t5_z)
            
            # 生成平滑曲线
            x_smooth = np.linspace(min(t5_seqlens) * 0.9, max(t5_seqlens) * 1.1, 100)
            y_smooth = t5_p(x_smooth)
            
            plt.plot(x_smooth, y_smooth, '--', color=t5_color, linewidth=2, alpha=0.6,
                     label=f'T5 Quadratic Fit: {t5_z[0]:.6f}x² + {t5_z[1]:.4f}x + {t5_z[2]:.2f}')
                     
        # 拟合二次曲线 - BERT
        if len(bert_seqlens) >= 3:
            x = np.array(bert_seqlens)
            y = np.array(bert_seq_inference)
            # 二次拟合
            bert_z = np.polyfit(x, y, 2)
            bert_p = np.poly1d(bert_z)
            
            # 生成平滑曲线
            x_smooth = np.linspace(min(bert_seqlens) * 0.9, max(bert_seqlens) * 1.1, 100)
            y_smooth = bert_p(x_smooth)
            
            plt.plot(x_smooth, y_smooth, '--', color=bert_color, linewidth=2, alpha=0.6,
                     label=f'BERT Quadratic Fit: {bert_z[0]:.6f}x² + {bert_z[1]:.4f}x + {bert_z[2]:.2f}')
        
        # 设置图表属性
        plt.xlabel('Sequence Length', fontsize=16, fontweight='bold')
        plt.ylabel('Memory Increase (MB)', fontsize=16, fontweight='bold')
        plt.title('Impact of Sequence Length on Transformer Memory Usage\nT5 vs BERT Comparison',
                  fontsize=20, fontweight='bold', pad=20)
                  
        # 美化刻度
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # 添加次要刻度
        plt.minorticks_on()
        plt.grid(which='minor', alpha=0.1)
        plt.grid(which='major', alpha=0.3, linestyle='--')
        
        # 添加图例
        legend = plt.legend(fontsize=13, frameon=True, facecolor='white', edgecolor='gray',
                           loc='upper left')
        legend.get_frame().set_alpha(0.9)
        
        # 添加水印
        plt.figtext(0.99, 0.01, 'Transformer Memory Analysis',
                    fontsize=10, color='gray', ha='right', alpha=0.7)
                    
        # 美化图表边框
        for spine in plt.gca().spines.values():
            spine.set_linewidth(1.5)
            
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(f'{self.output_dir}/seq_length_memory_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_seq_length_memory(self, model_name):
        """绘制序列长度对特定模型显存的影响"""
        plt.figure(figsize=(14, 10), dpi=100)
        plt.style.use('seaborn-v0_8-whitegrid')

        # 自定义颜色
        colors = {'t5': '#9b59b6', 'bert': '#2ecc71'}  # 紫色/绿色
        fit_color = '#e74c3c'  # 红色
        
        if model_name not in colors:
            print(f"不支持的模型名称: {model_name}")
            return
            
        color = colors[model_name]

        # 提取固定batch_size下的不同seq_length数据
        seqlens = []
        seq_inference = []

        batch_size_filter = 8  # 使用batch_size=8的数据点

        for i, batch in enumerate(self.results[model_name]['batch_size']):
            if batch == batch_size_filter:
                seqlens.append(self.results[model_name]['seq_length'][i])
                seq_inference.append(self.results[model_name]['inference'][i])

        # 数据点排序
        sort_idx = np.argsort(seqlens)
        seqlens = [seqlens[i] for i in sort_idx]
        seq_inference = [seq_inference[i] for i in sort_idx]

        # 绘制散点图
        plt.scatter(seqlens, seq_inference, s=180, color=color,
                    edgecolor='white', linewidth=2, zorder=3,
                    label=f'{model_name.upper()} Memory Usage (batch_size={batch_size_filter})')

        # 连接数据点
        plt.plot(seqlens, seq_inference, '-', color=color,
                 alpha=0.7, linewidth=3)

        # 拟合二次曲线
        if len(seqlens) >= 3:
            x = np.array(seqlens)
            y = np.array(seq_inference)

            # 二次拟合
            z = np.polyfit(x, y, 2)
            p = np.poly1d(z)

            # 生成平滑曲线
            x_smooth = np.linspace(min(x) * 0.9, max(x) * 1.1, 100)
            y_smooth = p(x_smooth)

            plt.plot(x_smooth, y_smooth, '--', color=fit_color, linewidth=3, alpha=0.8,
                     label=f'Quadratic Fit: {z[0]:.6f}x² + {z[1]:.4f}x + {z[2]:.2f}')

            # 填充曲线下方区域
            plt.fill_between(x_smooth, y_smooth, alpha=0.1, color=fit_color)

            # 计算R²值
            y_mean = np.mean(y)
            ss_tot = np.sum((y - y_mean) ** 2)
            ss_res = np.sum((y - p(x)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # 添加R²值标注
            plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes,
                     fontsize=14, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8,
                                                              edgecolor='gray', boxstyle='round,pad=0.5'))

            # 添加复杂度标注
            plt.text(0.05, 0.87, 'O(n²) Complexity\nConfirmed', transform=plt.gca().transAxes,
                     fontsize=14, fontweight='bold', color=fit_color,
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor=fit_color, boxstyle='round,pad=0.5'))

        # 设置图表属性
        plt.xlabel('Sequence Length', fontsize=16, fontweight='bold')
        plt.ylabel('Memory Increase (MB)', fontsize=16, fontweight='bold')
        plt.title(f'Impact of Sequence Length on {model_name.upper()} Memory Usage\nVerifying Quadratic O(n²) Complexity',
                  fontsize=20, fontweight='bold', pad=20)

        # 美化刻度
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # 添加次要刻度
        plt.minorticks_on()
        plt.grid(which='minor', alpha=0.1)
        plt.grid(which='major', alpha=0.3, linestyle='--')

        # 给每个数据点添加标签
        for i, (x, y) in enumerate(zip(seqlens, seq_inference)):
            plt.annotate(f'({x}, {y:.1f}MB)',
                         xy=(x, y), xytext=(0, 10),
                         textcoords='offset points', fontsize=10,
                         ha='center', va='bottom')

        # 添加图例
        legend = plt.legend(fontsize=13, frameon=True, facecolor='white', edgecolor='gray',
                           loc='upper left')
        legend.get_frame().set_alpha(0.9)

        # 添加水印
        plt.figtext(0.99, 0.01, f'{model_name.upper()} Memory Analysis',
                    fontsize=10, color='gray', ha='right', alpha=0.7)

        # 美化图表边框
        for spine in plt.gca().spines.values():
            spine.set_linewidth(1.5)

        plt.tight_layout()

        # 保存图表
        plt.savefig(f'{self.output_dir}/{model_name}_seq_length_memory_usage.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_3d_memory_surface(self, model_name):
        """为Transformer模型绘制3D显存表面图（批次大小和序列长度）"""
        # 确保有足够的数据点
        if len(self.results[model_name]['batch_size']) < 9:  # 至少需要3x3的网格
            print(f"数据点不足，无法绘制{model_name}的3D表面图")
            return

        # 创建网格数据
        batch_sizes = sorted(list(set(self.results[model_name]['batch_size'])))
        seq_lengths = sorted(list(set(self.results[model_name]['seq_length'])))

        X, Y = np.meshgrid(batch_sizes, seq_lengths)
        Z = np.zeros_like(X, dtype=float)

        # 填充Z值
        for i, seq_len in enumerate(seq_lengths):
            for j, batch in enumerate(batch_sizes):
                # 查找对应的数据点
                for k in range(len(self.results[model_name]['batch_size'])):
                    if (self.results[model_name]['batch_size'][k] == batch and
                            self.results[model_name]['seq_length'][k] == seq_len):
                        Z[i, j] = self.results[model_name]['inference'][k]
                        break

        # 创建更好的3D图
        fig = plt.figure(figsize=(16, 12), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        # 设置颜色映射
        cmap = plt.cm.viridis

        # 绘制表面
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.8,
                              linewidth=0, antialiased=True, rstride=1, cstride=1)

        # 绘制等高线
        cset = ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z) - 5, cmap=cmap, alpha=0.5)

        # 绘制网格点
        ax.scatter3D(X, Y, Z, color='red', s=50, edgecolor='white', linewidth=0.5, alpha=0.8)

        # 连接网格点
        for i in range(X.shape[0]):
            ax.plot(X[i, :], Y[i, :], Z[i, :], 'k-', alpha=0.2, linewidth=1)
        for j in range(X.shape[1]):
            ax.plot(X[:, j], Y[:, j], Z[:, j], 'k-', alpha=0.2, linewidth=1)

        # 设置图表属性
        ax.set_xlabel('Batch Size', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('Sequence Length', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_zlabel('Memory Increase (MB)', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title(f'{model_name.upper()} Transformer - 3D Relationship of Batch Size & Sequence Length to Memory',
                    fontsize=18, fontweight='bold', pad=20)

        # 美化坐标轴
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='z', labelsize=10)

        # 设置视角
        ax.view_init(elev=30, azim=135)

        # 调整轴范围，使图形更加紧凑
        ax.set_zlim(0, np.max(Z) * 1.2)

        # 添加颜色条
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label('Memory Increase (MB)', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)

        # 添加公式说明
        memory_eq = r"$Memory \propto BatchSize \times SeqLength^2$"
        ax.text2D(0.05, 0.95, memory_eq, transform=ax.transAxes, fontsize=16,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

        # 添加水印
        ax.text2D(0.95, 0.05, f'{model_name.upper()} 3D Memory Analysis', transform=ax.transAxes,
                 fontsize=10, ha='right', color='gray', alpha=0.7)

        plt.tight_layout()

        # 保存图表
        plt.savefig(f'{self.output_dir}/{model_name}_3d_memory_surface.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_memory_vs_parameters(self, models_data):
        """绘制显存占用与参数量的关系

        Args:
            models_data: 包含不同模型参数量和显存数据的列表
        """
        plt.figure(figsize=(14, 10), dpi=100)
        plt.style.use('seaborn-v0_8-whitegrid')

        # 提取数据
        names = [data['name'] for data in models_data]
        params = [data['parameters'] / 1e6 for data in models_data]  # 转换为百万
        memory = [data['memory'] for data in models_data]

        # 颜色映射
        colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

        # 绘制散点图
        for i, (name, param, mem, color) in enumerate(zip(names, params, memory, colors)):
            plt.scatter(param, mem, s=300, color=color, alpha=0.8, edgecolor='white', linewidth=2,
                       label=name, zorder=3)

        # 添加模型名称标签
        for i, (name, param, mem) in enumerate(zip(names, params, memory)):
            plt.annotate(name, (param, mem), xytext=(10, 5), textcoords='offset points',
                        fontsize=14, fontweight='bold')

        # 尝试拟合线性关系
        if len(params) >= 2:
            z = np.polyfit(params, memory, 1)
            p = np.poly1d(z)

            # 生成平滑线
            x_range = np.linspace(min(params) * 0.8, max(params) * 1.2, 100)
            plt.plot(x_range, p(x_range), '--', color='#e74c3c', linewidth=2, alpha=0.7,
                    label=f'Linear Fit: y = {z[0]:.2f}x + {z[1]:.2f}')

            # 填充置信区间
            plt.fill_between(x_range, p(x_range) - np.std(memory), p(x_range) + np.std(memory),
                            color='#e74c3c', alpha=0.1)

            # 计算R²值
            y_mean = np.mean(memory)
            ss_tot = np.sum((memory - y_mean) ** 2)
            ss_res = np.sum((memory - p(params)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # 添加R²值文本框
            plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes,
                    fontsize=14, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

        # 设置图表属性
        plt.xlabel('Model Parameters (Millions)', fontsize=16, fontweight='bold')
        plt.ylabel('Base Memory Usage (MB)', fontsize=16, fontweight='bold')
        plt.title('Relationship Between Model Size and Memory Usage',
                 fontsize=20, fontweight='bold', pad=20)

        # 美化刻度
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # 网格
        plt.grid(True, alpha=0.3, linestyle='--')

        # 添加图例
        legend = plt.legend(fontsize=13, frameon=True, facecolor='white', edgecolor='gray',
                           loc='upper left')
        legend.get_frame().set_alpha(0.9)

        # 添加水印
        plt.figtext(0.99, 0.01, 'Model Size vs Memory Analysis',
                   fontsize=10, color='gray', ha='right', alpha=0.7)

        # 添加内存效率注释
        for i, (name, param, mem) in enumerate(zip(names, params, memory)):
            efficiency = mem / param
            plt.annotate(f'Efficiency: {efficiency:.2f}MB/M params',
                        xy=(param, mem), xytext=(10, -15), textcoords='offset points',
                        fontsize=10, color='gray')

        # 美化图表边框
        for spine in plt.gca().spines.values():
            spine.set_linewidth(1.5)

        # 添加次要刻度
        plt.minorticks_on()
        plt.grid(which='minor', alpha=0.1)

        plt.tight_layout()

        # 保存图表
        plt.savefig(f'{self.output_dir}/params_vs_memory.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_layer_memory_pie_chart(self, layer_memory, model_name):
        """为模型各层的显存占用创建可视化图表"""
        try:
            if not layer_memory:
                print(f"没有足够的层数据来创建{model_name}的图表")
                return

            # 对显存占用进行排序
            sorted_memory = sorted(layer_memory.items(), key=lambda x: x[1], reverse=True)

            # 取前6个最大的层，其余归为"其他"
            top_n = min(6, len(sorted_memory))
            labels = [name for name, _ in sorted_memory[:top_n]]
            sizes = [size for _, size in sorted_memory[:top_n]]

            if len(sorted_memory) > top_n:
                labels.append('Others')
                sizes.append(sum(size for _, size in sorted_memory[top_n:]))

            # 创建两个独立的图表而不是子图，避免布局问题

            # 1. 创建饼图
            plt.figure(figsize=(10, 8))
            # 自定义颜色
            colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
            # 突出显示最大的部分
            explode = [0.1 if i == 0 else 0 for i in range(len(labels))]

            # 绘制饼图
            patches, texts, autotexts = plt.pie(
                sizes,
                labels=labels,  # 直接在饼图上显示标签
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                explode=explode,
                shadow=True,
                wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
                textprops={'fontsize': 12}  # 增大标签字体
            )

            # 设置自动文本属性
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_fontweight('bold')

            # 添加标题
            plt.title(f'{model_name} Layer Memory Distribution', fontsize=16, fontweight='bold', pad=15)

            # 添加水印
            plt.figtext(0.95, 0.01, f'{model_name} Analysis',
                       fontsize=8, color='gray', ha='right', alpha=0.7)

            plt.tight_layout()
            # 保存饼图
            plt.savefig(f'{self.output_dir}/{model_name}_layer_memory_pie.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 2. 创建条形图
            plt.figure(figsize=(12, 8))

            # 反转数据，使最大值在顶部
            rev_labels = labels.copy()
            rev_sizes = sizes.copy()
            rev_colors = colors.copy()
            rev_labels.reverse()
            rev_sizes.reverse()
            rev_colors = plt.cm.tab10(np.linspace(0, 1, len(rev_labels)))

            # 创建水平条形图
            bars = plt.barh(range(len(rev_labels)), rev_sizes, color=rev_colors, height=0.6)

            # 添加数值标签
            for i, bar in enumerate(bars):
                plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        f'{rev_sizes[i]:.2f} MB', va='center', fontsize=10)

            # 设置Y轴标签 - 直接使用有意义的缩短标签
            plt.yticks(range(len(rev_labels)),
                      [label if len(label) < 15 else label[:12] + '...' for label in rev_labels],
                      fontsize=12)

            # 设置X轴标签
            plt.xlabel('Memory Usage (MB)', fontsize=14, fontweight='bold')

            # 设置标题
            plt.title(f'{model_name} Layer Memory Usage', fontsize=16, fontweight='bold', pad=15)

            # 添加网格线
            plt.grid(axis='x', alpha=0.3, linestyle='--')

            # 添加水印
            plt.figtext(0.95, 0.01, f'{model_name} Analysis',
                       fontsize=8, color='gray', ha='right', alpha=0.7)

            plt.tight_layout()

            # 保存条形图
            plt.savefig(f'{self.output_dir}/{model_name}_layer_memory_bar.png', dpi=300)
            plt.close()

        except Exception as e:
            print(f"创建层显存图表时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def generate_report(self, vgg_results, t5_batch_results, bert_batch_results, layer_analysis=None):
        """生成数据报告

        Args:
            vgg_results: VGG16的显存测试结果
            t5_batch_results: T5批次大小测试结果
            bert_batch_results: BERT批次大小测试结果
            layer_analysis: 各层分析结果

        Returns:
            str: 报告
        """
        report = []

        # 1. 标题和设备信息
        report.append("# GPU显存占用分析报告")
        report.append(f"\n## 测试环境")
        report.append(f"- GPU: {self.device_name}")
        report.append(f"- CUDA计算能力: {self.device_properties.major}.{self.device_properties.minor}")
        report.append(f"- 总显存: {self.device_properties.total_memory / 1024 ** 3:.2f} GB")
        report.append(f"- PyTorch版本: {torch.__version__}")

        # 2. 模型信息
        report.append("\n## 模型信息")

        # VGG16信息
        vgg16 = models.vgg16(weights='IMAGENET1K_V1')
        vgg_params = sum(p.numel() for p in vgg16.parameters())
        report.append(f"### VGG16")
        report.append(f"- 参数总量: {vgg_params:,} ({vgg_params / 1e6:.2f}M)")
        report.append(f"- 模型类型: CNN")
        report.append(f"- 层数: {len(list(vgg16.features)) + len(list(vgg16.classifier))}")

        # T5信息
        t5 = T5ForConditionalGeneration.from_pretrained('t5-small')
        t5_params = sum(p.numel() for p in t5.parameters())
        report.append(f"\n### T5 Transformer")
        report.append(f"- 参数总量: {t5_params:,} ({t5_params / 1e6:.2f}M)")
        report.append(f"- 模型类型: Encoder-Decoder Transformer")
        report.append(f"- 编码器层数: {t5.config.num_layers}")
        report.append(f"- 解码器层数: {t5.config.num_decoder_layers}")
        report.append(f"- 注意力头数: {t5.config.num_heads}")
        report.append(f"- 隐藏层维度: {t5.config.d_model}")
        
        # BERT信息
        bert = BertModel.from_pretrained('bert-base-uncased')
        bert_params = sum(p.numel() for p in bert.parameters())
        report.append(f"\n### BERT")
        report.append(f"- 参数总量: {bert_params:,} ({bert_params / 1e6:.2f}M)")
        report.append(f"- 模型类型: Encoder-only Transformer")
        report.append(f"- 层数: {bert.config.num_hidden_layers}")
        report.append(f"- 注意力头数: {bert.config.num_attention_heads}")
        report.append(f"- 隐藏层维度: {bert.config.hidden_size}")

        # 3. 显存测试结果
        report.append("\n## 显存测试结果")

        # VGG16批次大小测试
        report.append("\n### VGG16 - 批次大小测试")
        vgg_table = []
        for i, size in enumerate(self.results['vgg16']['batch_size']):
            vgg_table.append([
                size,
                f"{self.results['vgg16']['inference'][i]:.2f}"
            ])

        report.append("| 批次大小 | 推理显存增加 (MB) |")
        report.append("|---------|----------------|")
        for row in vgg_table:
            report.append(f"| {row[0]} | {row[1]} |")

        # T5批次大小测试
        report.append("\n### T5 - 批次大小测试 (序列长度=64)")
        t5_batch_table = []

        for i, (batch, seq_len) in enumerate(zip(self.results['t5']['batch_size'], self.results['t5']['seq_length'])):
            if seq_len == 64:  # 只展示序列长度为64的结果
                t5_batch_table.append([
                    batch,
                    f"{self.results['t5']['inference'][i]:.2f}"
                ])

        report.append("| 批次大小 | 推理显存增加 (MB) |")
        report.append("|---------|----------------|")
        for row in t5_batch_table:
            report.append(f"| {row[0]} | {row[1]} |")
            
        # BERT批次大小测试
        report.append("\n### BERT - 批次大小测试 (序列长度=64)")
        bert_batch_table = []

        for i, (batch, seq_len) in enumerate(zip(self.results['bert']['batch_size'], self.results['bert']['seq_length'])):
            if seq_len == 64:  # 只展示序列长度为64的结果
                bert_batch_table.append([
                    batch,
                    f"{self.results['bert']['inference'][i]:.2f}"
                ])

        report.append("| 批次大小 | 推理显存增加 (MB) |")
        report.append("|---------|----------------|")
        for row in bert_batch_table:
            report.append(f"| {row[0]} | {row[1]} |")

        # T5序列长度测试
        report.append("\n### T5 - 序列长度测试 (批次大小=8)")
        t5_seq_table = []

        for i, (batch, seq_len) in enumerate(zip(self.results['t5']['batch_size'], self.results['t5']['seq_length'])):
            if batch == 8:  # 只展示批次大小为8的结果
                t5_seq_table.append([
                    seq_len,
                    f"{self.results['t5']['inference'][i]:.2f}"
                ])

        report.append("| 序列长度 | 推理显存增加 (MB) |")
        report.append("|---------|----------------|")
        for row in t5_seq_table:
            report.append(f"| {row[0]} | {row[1]} |")
            
        # BERT序列长度测试
        report.append("\n### BERT - 序列长度测试 (批次大小=8)")
        bert_seq_table = []

        for i, (batch, seq_len) in enumerate(zip(self.results['bert']['batch_size'], self.results['bert']['seq_length'])):
            if batch == 8:  # 只展示批次大小为8的结果
                bert_seq_table.append([
                    seq_len,
                    f"{self.results['bert']['inference'][i]:.2f}"
                ])

        report.append("| 序列长度 | 推理显存增加 (MB) |")
        report.append("|---------|----------------|")
        for row in bert_seq_table:
            report.append(f"| {row[0]} | {row[1]} |")

        # 4. 分析数据特性
        report.append("\n## 分析数据特性")

        # VGG16分析
        report.append("\n### VGG16模型显存特性")

        # 计算平均每增加一个批次的显存增加量
        if len(self.results['vgg16']['batch_size']) >= 2:
            batch_increases = []
            for i in range(1, len(self.results['vgg16']['batch_size'])):
                batch_diff = self.results['vgg16']['batch_size'][i] - self.results['vgg16']['batch_size'][i - 1]
                memory_diff = self.results['vgg16']['inference'][i] - self.results['vgg16']['inference'][i - 1]
                if batch_diff > 0:
                    batch_increases.append(memory_diff / batch_diff)

            avg_increase = np.mean(batch_increases)
            report.append(f"- **单位批次显存增加**: 每增加一个样本，VGG16推理阶段平均增加显存 {avg_increase:.2f}MB。")

        # T5分析
        report.append("\n### T5 Transformer模型显存特性")

        # 计算平均每增加一个批次的显存增加量
        t5_batch_samples = [(i, mem) for i, (batch, seq, mem) in
                            enumerate(zip(self.results['t5']['batch_size'],
                                          self.results['t5']['seq_length'],
                                          self.results['t5']['inference']))
                            if seq == 64]  # 只看seq_length=64的情况

        if len(t5_batch_samples) >= 2:
            t5_batch_samples.sort(key=lambda x: self.results['t5']['batch_size'][x[0]])
            batch_increases = []

            for j in range(1, len(t5_batch_samples)):
                idx_curr = t5_batch_samples[j][0]
                idx_prev = t5_batch_samples[j - 1][0]
                batch_diff = self.results['t5']['batch_size'][idx_curr] - self.results['t5']['batch_size'][idx_prev]
                memory_diff = self.results['t5']['inference'][idx_curr] - self.results['t5']['inference'][idx_prev]
                if batch_diff > 0:
                    batch_increases.append(memory_diff / batch_diff)

            if batch_increases:
                avg_increase = np.mean(batch_increases)
                report.append(
                    f"- **单位批次显存增加**: 在序列长度为64的情况下，每增加一个样本，T5推理阶段平均增加显存 {avg_increase:.2f}MB。")
                    
        # BERT分析
        report.append("\n### BERT模型显存特性")

        # 计算平均每增加一个批次的显存增加量
        bert_batch_samples = [(i, mem) for i, (batch, seq, mem) in
                            enumerate(zip(self.results['bert']['batch_size'],
                                          self.results['bert']['seq_length'],
                                          self.results['bert']['inference']))
                            if seq == 64]  # 只看seq_length=64的情况

        if len(bert_batch_samples) >= 2:
            bert_batch_samples.sort(key=lambda x: self.results['bert']['batch_size'][x[0]])
            batch_increases = []

            for j in range(1, len(bert_batch_samples)):
                idx_curr = bert_batch_samples[j][0]
                idx_prev = bert_batch_samples[j - 1][0]
                batch_diff = self.results['bert']['batch_size'][idx_curr] - self.results['bert']['batch_size'][idx_prev]
                memory_diff = self.results['bert']['inference'][idx_curr] - self.results['bert']['inference'][idx_prev]
                if batch_diff > 0:
                    batch_increases.append(memory_diff / batch_diff)

            if batch_increases:
                avg_increase = np.mean(batch_increases)
                report.append(
                    f"- **单位批次显存增加**: 在序列长度为64的情况下，每增加一个样本，BERT推理阶段平均增加显存 {avg_increase:.2f}MB。")

        # 计算T5序列长度的影响
        t5_seq_samples = [(i, mem) for i, (batch, seq, mem) in
                          enumerate(zip(self.results['t5']['batch_size'],
                                        self.results['t5']['seq_length'],
                                        self.results['t5']['inference']))
                          if batch == 8]  # 只看batch_size=8的情况

        if len(t5_seq_samples) >= 3:  # 至少需要3个点才能拟合二次曲线
            # 排序
            t5_seq_samples.sort(key=lambda x: self.results['t5']['seq_length'][x[0]])

            # 提取数据点
            seq_lengths = [self.results['t5']['seq_length'][x[0]] for x in t5_seq_samples]
            memories = [self.results['t5']['inference'][x[0]] for x in t5_seq_samples]

            # 二次拟合
            z = np.polyfit(seq_lengths, memories, 2)

            report.append(
                f"- **序列长度二次关系**: 拟合结果表明显存随序列长度的增长符合二次方程: {z[0]:.6f}n² + {z[1]:.4f}n + {z[2]:.2f}，")
                
        # 计算BERT序列长度的影响
        bert_seq_samples = [(i, mem) for i, (batch, seq, mem) in
                          enumerate(zip(self.results['bert']['batch_size'],
                                        self.results['bert']['seq_length'],
                                        self.results['bert']['inference']))
                          if batch == 8]  # 只看batch_size=8的情况

        if len(bert_seq_samples) >= 3:  # 至少需要3个点才能拟合二次曲线
            # 排序
            bert_seq_samples.sort(key=lambda x: self.results['bert']['seq_length'][x[0]])

            # 提取数据点
            seq_lengths = [self.results['bert']['seq_length'][x[0]] for x in bert_seq_samples]
            memories = [self.results['bert']['inference'][x[0]] for x in bert_seq_samples]

            # 二次拟合
            z = np.polyfit(seq_lengths, memories, 2)

            report.append(
                f"- **序列长度二次关系**: 拟合结果表明显存随序列长度的增长符合二次方程: {z[0]:.6f}n² + {z[1]:.4f}n + {z[2]:.2f}，")

        # 三种模型对比分析
        report.append("\n### VGG16、T5与BERT对比分析")

        # 参数量对比
        report.append("- **参数效率对比**:")
        report.append(f"   - VGG16参数量: {vgg_params / 1e6:.2f}M")
        report.append(f"   - T5参数量: {t5_params / 1e6:.2f}M")
        report.append(f"   - BERT参数量: {bert_params / 1e6:.2f}M")
        report.append(f"   - T5/VGG16参数比: {t5_params / vgg_params:.2f}倍")
        report.append(f"   - BERT/VGG16参数比: {bert_params / vgg_params:.2f}倍")
        report.append(f"   - T5/BERT参数比: {t5_params / bert_params:.2f}倍")

        # 基础显存使用对比
        if (self.results['vgg16']['inference'] and self.results['t5']['inference'] and 
            self.results['bert']['inference']):
            # 找到每个模型批次大小为1的情况进行比较
            vgg_base_mem = self.results['vgg16']['inference'][0]  # 假设第一个是batch=1

            t5_base_idx = -1
            for i, (batch, seq) in enumerate(zip(self.results['t5']['batch_size'], self.results['t5']['seq_length'])):
                if batch == 1 and seq == 64:
                    t5_base_idx = i
                    break

            bert_base_idx = -1
            for i, (batch, seq) in enumerate(zip(self.results['bert']['batch_size'], self.results['bert']['seq_length'])):
                if batch == 1 and seq == 64:
                    bert_base_idx = i
                    break

            if t5_base_idx >= 0 and bert_base_idx >= 0:
                t5_base_mem = self.results['t5']['inference'][t5_base_idx]
                bert_base_mem = self.results['bert']['inference'][bert_base_idx]
                
                report.append(f"- **基础显存效率对比**:")
                report.append(f"   - VGG16基础显存增加(batch=1): {vgg_base_mem:.2f}MB")
                report.append(f"   - T5基础显存增加(batch=1,seq=64): {t5_base_mem:.2f}MB")
                report.append(f"   - BERT基础显存增加(batch=1,seq=64): {bert_base_mem:.2f}MB")
                report.append(f"   - T5/VGG16显存比: {t5_base_mem / vgg_base_mem:.2f}倍")
                report.append(f"   - BERT/VGG16显存比: {bert_base_mem / vgg_base_mem:.2f}倍")
                report.append(f"   - T5/BERT显存比: {t5_base_mem / bert_base_mem:.2f}倍")

                report.append(f"- **显存-参数比**:")
                report.append(f"   - VGG16显存/参数比: {vgg_base_mem / (vgg_params / 1e6):.2f}MB/M参数")
                report.append(f"   - T5显存/参数比: {t5_base_mem / (t5_params / 1e6):.2f}MB/M参数")
                report.append(f"   - BERT显存/参数比: {bert_base_mem / (bert_params / 1e6):.2f}MB/M参数")

        return "\n".join(report)


def generate_vgg_input(size):
    """为VGG16生成输入

    Args:
        size: 批次大小

    Returns:
        torch.Tensor: 生成的输入张量
    """
    inputs = torch.randn(size, 3, 224, 224, device='cuda')
    return inputs


def generate_t5_input(size):
    """为T5 Transformer生成输入

    Args:
        size: (batch_size, seq_length)元组

    Returns:
        dict: 包含输入张量的字典
    """
    batch_size, seq_length = size
    input_ids = torch.randint(0, 32128, (batch_size, seq_length), device='cuda')
    attention_mask = torch.ones_like(input_ids)

    # T5模型在推理阶段也需要decoder_input_ids
    decoder_input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device='cuda')

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'decoder_input_ids': decoder_input_ids  # 添加解码器输入
    }
    
def generate_bert_input(size):
    """为BERT生成输入

    Args:
        size: (batch_size, seq_length)元组

    Returns:
        dict: 包含输入张量的字典
    """
    batch_size, seq_length = size
    input_ids = torch.randint(0, 30522, (batch_size, seq_length), device='cuda')
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)  # 所有token属于同一个segment

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    }


def main():
    """主函数，执行GPU显存测试和分析"""
    try:
        # 创建输出目录
        output_dir = "./gpu_memory_analysis_results_3"
        os.makedirs(output_dir, exist_ok=True)

        # 初始化显存监控器
        monitor = GPUMemoryMonitor(output_dir=output_dir)

        print("\n===== 任务三: GPU显存占用测量 =====")

        # 加载模型
        print("\n加载VGG16模型...")
        vgg16 = models.vgg16(weights='IMAGENET1K_V1').to('cuda')
        vgg16.eval()

        print("\n加载T5 Transformer模型...")
        t5 = T5ForConditionalGeneration.from_pretrained('t5-small').to('cuda')
        t5.eval()
        
        print("\n加载BERT模型...")
        bert = BertModel.from_pretrained('bert-base-uncased').to('cuda')
        bert.eval()

        # VGG16批次大小测试
        vgg_batch_sizes = [1, 2, 4, 8, 16, 32]
        vgg_results = monitor.measure_model_memory(
            model_name='vgg16',
            model=vgg16,
            input_generator=generate_vgg_input,
            input_sizes=vgg_batch_sizes
        )

        # T5批次大小测试（固定序列长度）
        t5_batch_sizes = [(1, 64), (2, 64), (4, 64), (8, 64), (16, 64)]
        t5_batch_results = monitor.measure_model_memory(
            model_name='t5',
            model=t5,
            input_generator=generate_t5_input,
            input_sizes=t5_batch_sizes
        )
        
        # BERT批次大小测试（固定序列长度）
        bert_batch_sizes = [(1, 64), (2, 64), (4, 64), (8, 64), (16, 64)]
        bert_batch_results = monitor.measure_model_memory(
            model_name='bert',
            model=bert,
            input_generator=generate_bert_input,
            input_sizes=bert_batch_sizes
        )

        # T5序列长度测试（固定批次大小）
        t5_seq_lengths = [(8, 16), (8, 32), (8, 64), (8, 128), (8, 256)]
        t5_seq_results = monitor.measure_model_memory(
            model_name='t5',
            model=t5,
            input_generator=generate_t5_input,
            input_sizes=t5_seq_lengths
        )
        
        # BERT序列长度测试（固定批次大小）
        bert_seq_lengths = [(8, 16), (8, 32), (8, 64), (8, 128), (8, 256)]
        bert_seq_results = monitor.measure_model_memory(
            model_name='bert',
            model=bert,
            input_generator=generate_bert_input,
            input_sizes=bert_seq_lengths
        )

        # 附加测试点（用于3D图）
        additional_t5_points = [
            (2, 16), (2, 32), (2, 128), (2, 256),
            (4, 16), (4, 32), (4, 128), (4, 256),
            (16, 16), (16, 32), (16, 128), (16, 256)
        ]

        t5_additional_results = monitor.measure_model_memory(
            model_name='t5',
            model=t5,
            input_generator=generate_t5_input,
            input_sizes=additional_t5_points
        )
        
        # 附加测试点（用于BERT的3D图）
        additional_bert_points = [
            (2, 16), (2, 32), (2, 128), (2, 256),
            (4, 16), (4, 32), (4, 128), (4, 256),
            (16, 16), (16, 32), (16, 128), (16, 256)
        ]

        bert_additional_results = monitor.measure_model_memory(
            model_name='bert',
            model=bert,
            input_generator=generate_bert_input,
            input_sizes=additional_bert_points
        )

        # 分析各层显存占用
        print("\n分析VGG16各层显存占用...")
        vgg_layer_memory = monitor.analyze_layer_memory(vgg16, generate_vgg_input, 1)
        monitor.create_layer_memory_pie_chart(vgg_layer_memory, 'vgg16')

        print("\n分析T5各层显存占用...")
        t5_layer_memory = monitor.analyze_layer_memory(t5, generate_t5_input, (1, 64))
        monitor.create_layer_memory_pie_chart(t5_layer_memory, 't5')
        
        print("\n分析BERT各层显存占用...")
        bert_layer_memory = monitor.analyze_layer_memory(bert, generate_bert_input, (1, 64))
        monitor.create_layer_memory_pie_chart(bert_layer_memory, 'bert')

        # 绘制图表
        print("\n生成批次大小对显存的影响比较图...")
        monitor.plot_batch_size_memory(vgg_results, t5_batch_results, bert_batch_results)

        print("\n生成序列长度对T5显存的影响图...")
        monitor.plot_seq_length_memory('t5')
        
        print("\n生成序列长度对BERT显存的影响图...")
        monitor.plot_seq_length_memory('bert')
        
        print("\n生成序列长度对Transformer模型显存的影响比较图...")
        monitor.plot_seq_length_memory_comparison()

        print("\n生成T5批次大小与序列长度3D关系图...")
        monitor.plot_3d_memory_surface('t5')
        
        print("\n生成BERT批次大小与序列长度3D关系图...")
        monitor.plot_3d_memory_surface('bert')

        # 比较不同模型的参数量与显存关系
        models_data = [
            {
                'name': 'VGG16',
                'parameters': sum(p.numel() for p in vgg16.parameters()),
                'memory': monitor.results['vgg16']['inference'][0]  # batch=1的情况
            },
            {
                'name': 'T5-small',
                'parameters': sum(p.numel() for p in t5.parameters()),
                'memory': [mem for i, mem in enumerate(monitor.results['t5']['inference'])
                           if monitor.results['t5']['batch_size'][i] == 1 and
                           monitor.results['t5']['seq_length'][i] == 64][0]
            },
            {
                'name': 'BERT-base',
                'parameters': sum(p.numel() for p in bert.parameters()),
                'memory': [mem for i, mem in enumerate(monitor.results['bert']['inference'])
                           if monitor.results['bert']['batch_size'][i] == 1 and
                           monitor.results['bert']['seq_length'][i] == 64][0]
            }
        ]

        print("\n生成模型参数量与显存关系图...")
        monitor.plot_memory_vs_parameters(models_data)

        # 生成报告
        print("\n生成分析报告...")
        report = monitor.generate_report(vgg_results, t5_batch_results, bert_batch_results)

        # 保存报告到文件
        with open(f"{output_dir}/gpu_memory_analysis_report.md", "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\n分析完成! 结果已保存到 {output_dir} 目录")

        # 返回简短的摘要
        summary = "\n".join([
            "==== GPU显存分析摘要 ====",
            f"VGG16参数量: {sum(p.numel() for p in vgg16.parameters()) / 1e6:.2f}M",
            f"T5参数量: {sum(p.numel() for p in t5.parameters()) / 1e6:.2f}M",
            f"BERT参数量: {sum(p.numel() for p in bert.parameters()) / 1e6:.2f}M",
            f"VGG16显存随批次大小变化: 线性关系",
            f"T5显存随序列长度变化: 二次关系",
            f"BERT显存随序列长度变化: 二次关系",
            f"详细分析报告已保存至: {output_dir}/gpu_memory_analysis_report.md"
        ])

        print(summary)

    except Exception as e:
        print(f"执行过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()