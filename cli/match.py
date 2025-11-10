# cli/match.py
import argparse, json, cv2
from fusion.parallel_service import match_frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to probe image")
    parser.add_argument("--Tnh", type=float, default=30, help="NeuralHash reject threshold")
    parser.add_argument("--Thdic", type=float, default=3100, help="HDIC reject threshold")
    parser.add_argument("--w_nh", type=float, default=0.4, help="Fusion weight for NH")
    parser.add_argument("--w_hdic", type=float, default=0.6, help="Fusion weight for HDIC")
    parser.add_argument("--fused_th", type=float, default=0.75, help="Final fused threshold")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"[ERROR] cannot read image: {args.image}")
        return

    res = match_frame(
        img,
        Tnh=args.Tnh,
        Thdic=args.Thdic,
        w_nh=args.w_nh,
        w_hdic=args.w_hdic,
        fused_th=args.fused_th,
    )
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
