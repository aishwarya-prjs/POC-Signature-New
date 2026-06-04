
import argparse
from loguru import logger
from rich.console import Console

console = Console()

def main(args):
    from src.detection.detector import SignatureDetector

    console.print("[bold cyan]Starting YOLOv8 training...[/bold cyan]")
    console.print(f"  Epochs: {args.epochs} | Batch: {args.batch} | Device: {args.device}")

    detector = SignatureDetector(
        model_path=args.model_out,
        device=args.device,
    )
    results = detector.train(
        data_config=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
    )
    console.print(f"\n[green]✓[/green] Training complete!")
    console.print(f"Best weights saved to: [cyan]{args.model_out}[/cyan]")
    console.print("\nNext step: [cyan]python scripts/run_pipeline.py --image path/to/image.jpg[/cyan]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on signature dataset")
    parser.add_argument("--data", default="configs/dataset.yaml", help="Dataset config YAML")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="cpu or 0 (GPU)")
    parser.add_argument("--model-out", default="models/signature_yolov8.pt")
    main(parser.parse_args())
