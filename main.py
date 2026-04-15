import argparse
import os
import json
import warnings
warnings.filterwarnings('ignore')

from config import DatasetConfig, ModelConfig
from dataset import create_dataloaders
from train_eval import get_evaluator

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-modal PatchCore Anomaly Detection')
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--raw_data_root', type=str, default='./data/MVTec3D-AD')
    parser.add_argument('--categories', nargs='+', default=['dowel'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--f_coreset', type=float, default=0.01)
    parser.add_argument('--visualize', action='store_true', help='Save visualization images')
    parser.add_argument('--save_model', action='store_true', help='Save the trained feature bank/model')
    parser.add_argument('--load_model', action='store_true', help='Load the feature bank/model instead of training')
    return parser.parse_args()

def main():
    """
    程序总执行入口主函数。
    读取配置、构建对应类别专用 Evaluator 检测模型，并运行评估与可视化流程。
    """
    args = parse_args()
    
    dataset_config = DatasetConfig(
        raw_data_root=args.raw_data_root,
        categories=args.categories,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        test_batch_size=args.batch_size
    )
    
    model_config = ModelConfig(
        f_coreset=args.f_coreset
    )
    
    print("\n" + "="*60)
    print("Multi-modal PatchCore (ResNet18) Evaluation")
    print("="*60)

    for category in args.categories:
        print(f"\nProcessing category: {category}")
        evaluator = get_evaluator(category, config=model_config)
        train_loader, _, test_loader = create_dataloaders(dataset_config, [category])
        
        ckpt_dir = "checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{category}_model.pth")
        
        if args.load_model and os.path.exists(ckpt_path):
            print(f"Loading cached model from {ckpt_path}...")
            evaluator.load_model(ckpt_path)
        else:
            evaluator.build_feature_bank(train_loader)
            if args.save_model:
                evaluator.save_model(ckpt_path)
        
        results = evaluator.compute_anomaly_scores(test_loader)
        
        if args.visualize:
            print(f"Generating visualizations to ./visualizations/{category} ...")
            evaluator.visualize_results(test_loader, f"./visualizations/{category}", num_samples=5)
        
        print("\nEvaluation Results:")
        print("-" * 40)
        print(f"Sample-level AUROC: {results['sample_level']['auroc']:.4f}")
        print(f"Sample-level AP: {results['sample_level']['ap']:.4f}")
        if 'point_level' in results:
            print(f"Point-level AUROC: {results['point_level']['auroc']:.4f}")
            print(f"Point-level AP: {results['point_level']['ap']:.4f}")

if __name__ == "__main__":
    main()
