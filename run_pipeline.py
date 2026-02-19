"""
Simple script to run the pipeline from command line
"""

import argparse
import sys
import os
from pipeline_orchestrator import CrackAnalysisPipeline

def main():
    parser = argparse.ArgumentParser(description="Run crack analysis pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--yolo", type=str, 
                       default="Crack_Detection_YOLO/crack_yolo_train/weights/best.pt",
                       help="Path to YOLO model")
    parser.add_argument("--seg", type=str,
                       default="Masking_and_Classification_model/pretrained_net_G.pth",
                       help="Path to segmentation model")
    parser.add_argument("--class", type=str,
                       default="Masking_and_Classification_model/crack_orientation_classifier.h5",
                       dest="class_model",
                       help="Path to classification model")
    parser.add_argument("--rag-dir", type=str,
                       default="Rag_and_Reasoning/crack_analysis_rag/data",
                       help="Path to RAG data directory")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG analysis")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return 1
    
    # Initialize pipeline
    print("Initializing pipeline...")
    try:
        pipeline = CrackAnalysisPipeline(
            yolo_model_path=args.yolo,
            segmentation_model_path=args.seg,
            classification_model_path=args.class_model,
            rag_data_dir=args.rag_dir if not args.no_rag else None,
            device=args.device
        )
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Process image
    print(f"\nProcessing image: {args.image}")
    try:
        results = pipeline.process_image(
            args.image,
            use_rag=not args.no_rag,
            save_intermediate=True,
            output_dir=args.output
        )
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        summary = pipeline.get_summary(results)
        print(summary)
        
        # Save summary to file
        summary_path = os.path.join(args.output, "summary.txt")
        os.makedirs(args.output, exist_ok=True)
        with open(summary_path, 'w') as f:
            f.write(summary)
        print(f"\nSummary saved to: {summary_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())




