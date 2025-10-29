"""Run the ErgonomicXAI prototype on a downloaded subset of images.

This script runs `validate.py`-like steps but accepts parameters for how many
images to process and reduces SHAP nsamples for faster iteration.

Usage:
    python scripts/run_prototype_on_subset.py --limit 10 --nsamples 30
"""
import argparse
import sys
from pathlib import Path
import numpy as np
try:
    import cv2
except Exception:
    cv2 = None

# matplotlib and plotting are optional; import later to avoid heavy/ABI-sensitive imports
plt = None

# make sure project root is on sys.path so local imports work when scripts are run directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# We'll lazy-import project modules inside main() when not running in mock mode.


def main(limit: int = 10, nsamples: int = 30, outdir: Path = None, mock: bool = False, explain_mode: str = 'temporal'):
    root = Path(__file__).parent.parent
    image_dir = root / "data" / "images"
    if outdir is None:
        out_dir = root / "validation_results" / "run_subset"
    else:
        out_dir = root / "validation_results" / outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(image_dir.glob("*.jpg"))[:limit]
    if not images:
        print("No images found in data/images/ — run get_coco_wholebody_subset.py first")
        return

    print(f"Processing {len(images)} images with SHAP nsamples={nsamples} (mock={mock}) ...")

    # If cv2 or mediapipe aren't available, we support a mock mode where we
    # fabricate simple poses so the rest of the pipeline can run for testing.
    # Lazy import project modules to avoid importing heavy deps when running in mock mode
    if not mock:
        from src.pose_extraction import PoseExtractor
        from src.risk_calculation import RiskCalculator
        from src.temporal_model import ErgonomicTemporalModel
        from src.explainability import ExplainabilityModule
        from src.advice import generate_advice
        # visualization is optional (cv2/matplotlib heavy); import lazily
        try:
            from src.visualize import VisualizationModule
        except Exception:
            VisualizationModule = None
        extractor = None
        try:
            extractor = PoseExtractor()
        except Exception:
            extractor = None
        risk_calc = RiskCalculator()
        temp_model = ErgonomicTemporalModel()
        temp_model.train_with_dummy_data(epochs=1)
        explainer = ExplainabilityModule(temp_model, risk_calc)
    else:
        # Simple mock implementations for fast, dependency-free runs
        class RiskCalculator:
            def calculate_reba_score(self, landmarks):
                # return mid-level risk and simple breakdown
                return 5.0, {'trunk': 0.4, 'arms': 0.4, 'legs': 0.2}

        class ErgonomicTemporalModel:
            def __init__(self):
                self.is_trained = True
            def train_with_dummy_data(self, *a, **k):
                pass

        class ExplainabilityModule:
            def __init__(self, model, rc):
                pass
            def generate_explanation(self, flat, nsamples=20):
                # deterministic pseudo-explanation from flat
                s = float(np.mean(flat))
                parts = {'trunk': 0.33, 'arms': 0.33, 'legs': 0.34}
                text = 'Mock explanation — use real run for accurate SHAP attributions.'
                return parts, text
            def generate_explanation_frame_only(self, flat, nsamples=20, background_samples=100):
                return self.generate_explanation(flat, nsamples=nsamples)

        extractor = None
        risk_calc = RiskCalculator()
        temp_model = ErgonomicTemporalModel()
        explainer = ExplainabilityModule(temp_model, risk_calc)

    for img_path in images:
        # If cv2 is available, try to read the image; otherwise skip image data.
        img = None
        if cv2 is not None:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  - could not read {img_path.name}")
                continue

        if extractor is None or mock:
            # Mock pose: produce a deterministic synthetic landmark set so downstream
            # code can compute a risk and generate an explanation. We'll make a
            # flattened array of 33 landmarks (MediaPipe Pose has 33) with x,y,z.
            lm_count = 33
            # simple pattern based on filename to vary poses
            seed = sum(ord(c) for c in img_path.name) % 100
            rng = np.random.RandomState(seed)
            flat = rng.rand(lm_count * 3)
            # Mock world landmarks container for risk calc: we supply a lightweight object
            class _L:
                def __init__(self, x, y, z):
                    self.x = x
                    self.y = y
                    self.z = z

            class _R:
                def __init__(self, flat, lm_count):
                    self.pose_landmarks = type('P', (), {'landmark': [ _L(*flat[i*3:(i+1)*3]) for i in range(lm_count) ]})
                    self.pose_world_landmarks = self.pose_landmarks

            results = _R(flat, lm_count)
            reba, breakdown = risk_calc.calculate_reba_score(results.pose_world_landmarks)
            if explain_mode == 'frame':
                parts, text = explainer.generate_explanation_frame_only(flat, nsamples=nsamples)
            else:
                parts, text = explainer.generate_explanation(flat, nsamples=nsamples)
        else:
            results, _ = extractor.extract_pose(img)
            if not results.pose_world_landmarks:
                print(f"  - no pose in {img_path.name}")
                continue
            reba, breakdown = risk_calc.calculate_reba_score(results.pose_world_landmarks)
            flat = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
            if explain_mode == 'frame':
                parts, text = explainer.generate_explanation_frame_only(flat, nsamples=nsamples)
            else:
                parts, text = explainer.generate_explanation(flat, nsamples=nsamples)
        print(f"{img_path.name}: REBA={reba:.2f} | top={max(parts, key=parts.get)} | parts={parts}")

        # save a small JSON summary and a bar chart for quick inspection
        try:
            import json
            advice_det = generate_advice(float(reba), breakdown, parts, getattr(results, 'pose_world_landmarks', None) if not mock else None)
            primary_shap = max(parts, key=parts.get) if parts else None
            # primary_det: highest from breakdown (deterministic) if available
            primary_det = None
            try:
                if breakdown and isinstance(breakdown, dict):
                    primary_det = max(breakdown, key=breakdown.get)
            except Exception:
                primary_det = None

            summary = {
                'image': img_path.name,
                'reba': float(reba),
                'parts': parts,
                'advice_shap': text,
                'advice_det': advice_det,
                'primary_shap': primary_shap,
                'primary_det': primary_det,
                'advice_conflict': (primary_shap != primary_det) if (primary_shap is not None and primary_det is not None) else False
            }
            (out_dir / f"{img_path.stem}.json").write_text(json.dumps(summary, indent=2))
            # optionally create an overlay visualization if possible
            try:
                if VisualizationModule is not None and not mock and cv2 is not None and img is not None:
                    try:
                        viz = VisualizationModule(
                            img,
                            results.pose_landmarks if hasattr(results, 'pose_landmarks') else results,
                            reba,
                            breakdown,
                            [[float(0.0)]],
                            parts,
                            summary.get('advice_shap', '')
                        )
                        viz_path = out_dir / f"{img_path.stem}_viz.png"
                        viz.generate_visualization(str(viz_path))
                    except Exception:
                        pass
            except Exception:
                pass
            # Try plotting if matplotlib is available; import lazily to avoid heavy imports
            try:
                global plt
                if plt is None:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as _plt
                    plt = _plt
                plt.figure(figsize=(4,3))
                plt.bar(list(parts.keys()), list(parts.values()), color=['#4c72b0', '#dd8452', '#55a868'])
                plt.ylim(0,1)
                plt.title(f"{img_path.name} parts importance")
                plt.tight_layout()
                plt.savefig(out_dir / f"{img_path.stem}_parts.png")
                plt.close()
            except Exception:
                # plotting failed (likely due to numpy/matplotlib mismatch); continue without images
                pass
        except Exception as e:
            print(f"  - warning: could not save artifacts for {img_path.name}: {e}")

    print("Done. Outputs saved to:")
    print(f"  - {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--nsamples', type=int, default=30)
    parser.add_argument('--mock', action='store_true', help='Run in mock mode without cv2/mediapipe')
    parser.add_argument('--outdir', type=str, default=None, help='Optional output directory under validation_results')
    parser.add_argument('--explain-mode', type=str, choices=['temporal', 'frame'], default='temporal',
                        help='Choose explanation mode: temporal (default) uses the temporal model; frame uses direct REBA frame-attributions')
    args = parser.parse_args()
    # pass outdir Path if provided
    outdir_arg = Path(args.outdir) if args.outdir else None
    main(limit=args.limit, nsamples=args.nsamples, outdir=outdir_arg, mock=args.mock, explain_mode=args.explain_mode)
