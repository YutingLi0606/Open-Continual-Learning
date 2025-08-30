import argparse
import torch


def parse_arguments():
    parser = argparse.ArgumentParser()

    # ==================== 基础超参数 ====================
    parser.add_argument("--model", type=str, default="ViT-B/16", help="CLIP model name")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--batch-size-eval", type=int, default=128, help="Evaluation batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--ls", type=float, default=0.0, help="Label smoothing")
    parser.add_argument("--warmup_length", type=int, default=100, help="Warmup iterations")
    parser.add_argument("--beta2", type=float, default=0.999, help="AdamW beta2 parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # ==================== 训练控制 ====================
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--iterations", type=int, default=None, help="Number of iterations")
    parser.add_argument("--eval-interval", type=int, default=None, help="Evaluation interval")
    parser.add_argument("--loss-interval", type=int, default=1000, help="Loss logging interval")
    parser.add_argument("--eval-every-epoch", action="store_true", help="Evaluate every epoch")
    parser.add_argument("--eval-only", action="store_true", help="Evaluation only mode")

    # ==================== 实验设置 ====================
    parser.add_argument(
        "--method",
        type=str,
        default="finetune",
        choices=["finetune", "lwf", "ZSCL", "icarl", "rail", "moe"],
        help="Training method to use"
    )
    parser.add_argument(
        "--train-mode",
        type=str,
        default="whole",
        choices=["whole", "text", "image", "image-fc", "image-fc-fixed", "fc", "adapter"],
        help="Training mode"
    )
    parser.add_argument("--data-location", type=str, default="/data_ssd/mtil_datasets", help="Dataset location")
    parser.add_argument("--train-dataset", default=None, help="Training dataset")
    parser.add_argument("--eval-datasets", default=None, type=lambda x: x.split(","), help="Evaluation datasets")
    parser.add_argument("--text-datasets", default=None, type=lambda x: x.split(","), help="Text datasets")
    parser.add_argument("--template", type=str, default=None, help="Prompt template")

    # ==================== 模型保存与加载 ====================
    parser.add_argument("--save", type=str, default=None, help="Save path")
    parser.add_argument("--load", type=str, default=None, help="Load model path")
    parser.add_argument("--load_federate", default=None, type=lambda x: x.split(","), help="Load federated models")
    parser.add_argument("--load_autochooser", type=str, default=None, help="Load auto-chooser path")

    # ==================== 模型控制参数 ====================
    # 权重平均相关
    parser.add_argument("--fair", action="store_true", help="Fair training")
    parser.add_argument("--we", action="store_true", help="Weight ensemble")
    parser.add_argument("--we_wise", action="store_true", help="WiSE weight ensemble")
    parser.add_argument("--we_wise_alpha", type=float, default=0.98, help="WiSE ensemble alpha")
    parser.add_argument("--moving_avg", action="store_true", help="Moving average")
    parser.add_argument("--avg_freq", type=int, default=100, help="Average frequency")
    parser.add_argument("--mv_avg_decay", type=float, default=0.999, help="Moving average decay")
    parser.add_argument(
        "--mv_avg_model",
        type=str,
        default="n",
        choices=["n", "t", "zeroshot"],
        help="Moving average model type"
    )
    
    # 正则化相关
    parser.add_argument("--l2", type=float, default=0, help="L2 regularization weight")
    parser.add_argument("--fc-init", action="store_true", help="Reinitialize FC layers")
    parser.add_argument("--fc-setnone", action="store_true", help="Set FC layers to none")
    parser.add_argument("--dataset-shift", action="store_true", help="Shift dataset")
    parser.add_argument("--n_class", type=int, default=10, help="Number of classes")

    # ==================== ZSCL 方法参数 ====================
    parser.add_argument("--ref_wise_alpha", type=float, default=0.8, help="WiSE zeroshot reference alpha")
    parser.add_argument("--ref-wise", action="store_true", help="Use WiSE zeroshot reference")
    parser.add_argument("--ref-dataset", default=None, help="Reference dataset")
    parser.add_argument("--ref-model", type=str, default=None, help="Reference model path")
    parser.add_argument("--ref-sentences", default=None, help="Reference sentences dataset")
    parser.add_argument("--T", type=float, default=2.0, help="Temperature for distillation loss")
    parser.add_argument("--num", type=float, default=64, help="Number of reference samples")

    # ==================== iCaRL 方法参数 ====================
    parser.add_argument("--dataset_order", default=None, type=lambda x: x.split(","), help="Dataset order")
    parser.add_argument("--memory_size", type=int, default=10000, help="Memory size")

    # ==================== 损失函数控制 ====================
    parser.add_argument("--weight_adjust", action="store_true", help="Adjust loss weights")
    parser.add_argument("--feature_mse", action="store_true", help="Feature MSE loss")
    parser.add_argument("--image_loss", action="store_true", help="Image loss")
    parser.add_argument("--text_loss", action="store_true", help="Text loss")
    parser.add_argument("--ablation_loss_2", action="store_true", help="Ablation loss 2")

    # ==================== WiSE 相关 ====================
    parser.add_argument("--wise_merge", action="store_true", help="Use wise merge during training")
    parser.add_argument("--wise_ft", action="store_true", help="Use wise fine-tuning during evaluation")
    parser.add_argument("--wise_ft_model", type=str, default="n", choices=["n", "zeroshot"], help="WiSE FT model type")
    parser.add_argument("--wise_ft_alpha", type=float, default=0.8, help="WiSE FT alpha")

    # ==================== 日志与存储 ====================
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--results-db", type=str, default="results.jsonl", help="Results database path")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory")

    # ==================== 模型冻结 ====================
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze image encoder")
    parser.add_argument("--freeze-fc", type=int, default=0, help="Freeze FC layers")

    # ==================== LWF 方法参数 ====================
    parser.add_argument("--lwf", action="store_true", help="Use LWF")
    parser.add_argument("--basic_model_load", type=lambda x: x.split(","), default=None, help="Load basic models")
    parser.add_argument("--fc_load", type=lambda x: x.split(","), default=None, help="Load FC layers")
    parser.add_argument("--keep_old_heads", type=int, default=0, help="Keep old heads")

    # ==================== 基线方法 ====================
    parser.add_argument("--baseline", action="store_true", help="Use baseline method")

    # ==================== TRIO 方法参数 ====================
    parser.add_argument("--trio", action="store_true", help="Use TRIO")
    parser.add_argument("--control-dataset", default=None, help="Control dataset")
    parser.add_argument("--control-dataset-add", default=None, help="Additional control dataset")
    parser.add_argument("--noise", action="store_true", help="Use random noise regularization")
    parser.add_argument("--rff", action="store_true", help="Use RFF")

    # ==================== WiSE-FT 参数 ====================
    parser.add_argument("--alpha", default=0.5, type=float, help="WiSE alpha")
    parser.add_argument("--fisher", type=lambda x: x.split(","), default=None, help="Fisher information")
    parser.add_argument("--fisher_floor", type=float, default=1e-8, help="Fisher floor")

    # ==================== MoE 方法参数 ====================
    parser.add_argument("--ffn_num", type=int, default=64, help="FFN dimension")
    parser.add_argument("--ffn_adapt", action="store_true", help="Use adapter")
    parser.add_argument("--ffn_option", type=str, default="parallel", help="FFN option")
    parser.add_argument(
        "--ffn_adapt_where",
        type=str,
        default="AdapterDoubleEncoder",
        choices=["AdapterImageEncoder", "AdapterDoubleEncoder"],
        help="Adapter location"
    )
    parser.add_argument("--apply_moe", action="store_true", help="Use MoE")
    parser.add_argument("--repeat_train", action="store_true", help="Repeat training")
    parser.add_argument("--task_id", type=int, default=-1, help="Task ID")
    parser.add_argument("--multi_experts", action="store_true", help="Use multiple experts")
    parser.add_argument("--experts_num", type=int, default=2, help="Number of experts")
    parser.add_argument("--is_train", action="store_true", help="Training mode for router")
    parser.add_argument("--frozen", action="store_true", help="Frozen parameters")
    parser.add_argument("--autorouter", action="store_true", help="Use auto router")
    parser.add_argument("--task_num", type=int, default=11, help="Maximum number of tasks")
    parser.add_argument("--threshold", type=float, help="Zero-shot threshold")
    parser.add_argument("--non_text", action="store_true", help="Do not use text encoder")
    parser.add_argument("--frozen-path", type=str, default="frozen_list", help="Frozen list path")
    parser.add_argument("--few_shot", type=int, default=-1, help="Few-shot number")
    parser.add_argument("--topk", type=int, default=2, help="Top-k selection")
    parser.add_argument("--train_chooser", action="store_true", help="Train auto-chooser")

    # ==================== RAIL 方法参数 ====================
    # Primal RAIL
    parser.add_argument("--regularization", type=float, default=0.1, help="Regularization parameter for primal RAIL")
    parser.add_argument("--fusion_expansion", type=int, default=1, help="Fusion expansion")
    parser.add_argument("--hidden_dim", type=int, default=15000, help="Hidden dimension for feature expansion")
    
    # Dual RAIL (KRR)
    parser.add_argument("--krr_gamma", type=float, default=0.1, help="KRR RBF kernel parameter")
    parser.add_argument("--krr_lambda", type=float, default=0.001, help="KRR regularization parameter")
    
    # 通用 RAIL 参数
    parser.add_argument("--fusion_weight", type=float, default=0.8, help="Fusion weight between zero-shot and RAIL")
    parser.add_argument("--augmentation_time", type=int, default=1, help="Data augmentation times")
    parser.add_argument("--eval_adapter", type=int, default=1, help="Evaluate adapter")
    parser.add_argument("--eval_last", type=int, default=0, help="Evaluate only last task")
    parser.add_argument(
        "--rail_version",
        type=str,
        default="dual",
        choices=["dual", "primal"],
        help="RAIL version: dual (KRR) or primal (analytic adaptation)"
    )
    parser.add_argument(
        "--dataset_sequence", 
        type=str, 
        nargs="+",
        default=["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers", "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"],
        help="Dataset sequence for multi-domain training"
    )

    # 解析参数
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # 参数验证
    assert (
        args.epochs is None or args.iterations is None
    ), "Cannot specify both epoch and iterations."
    assert (
        args.eval_interval is None or not args.eval_every_epoch
    ), "Cannot specify both eval_interval and eval_every_epoch."

    return args