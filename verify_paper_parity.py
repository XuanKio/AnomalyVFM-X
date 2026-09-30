"""Compare the web runtime against unmodified authors' single-image statements.

Uses the same cached official HF weights for both runs (not an independent PKL
checkpoint comparison). CUDA BF16 required. No dataset/model downloads.
"""
import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

from paper_inference import PaperRuntime, predict_single_image
from demo_fast import MODEL_ID, MODEL_CACHE_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for parity with the original CUDA script.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    original_path = root / 'predict_single_image.py'
    source = original_path.read_text(encoding='utf-8')
    original = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    start = next(i for i, n in enumerate(original.body)
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id == 'img_transform')
    # Execute original transform, autocast, forward, pooling and save argument
    # verbatim. Avoid importing the training/evaluation dependencies in test.py.
    body = ast.Module(body=original.body[start:], type_ignores=[])
    authors_code = compile(ast.fix_missing_locations(body), str(original_path), 'exec')
    runtime = PaperRuntime()
    config_path = Path(hf_hub_download(MODEL_ID, 'config.json',
                                     cache_dir=MODEL_CACHE_DIR, local_files_only=True))
    assert all(p.dtype == torch.float32 for p in runtime.model.parameters())
    samples = []
    for name in ('hazelnut_normal.png', 'hazelnut_defective.png', 'bottle_defective.png'):
        path = root / 'demo_images' / name
        captured = {}

        def capture(mask, *unused, **kwargs):
            captured['mask'] = mask.detach().cpu()

        scope = dict(torch=torch, nn=torch.nn, Image=Image,
                     model=runtime.model.model, decoder=runtime.model.decoder,
                     predictor=runtime.model.predictor, feat_size=runtime.model.feat_size,
                     args=SimpleNamespace(mean_kernel_size=5, image_path=str(path)),
                     save_predictions_with_paths=capture)
        exec(authors_code, scope)
        with Image.open(path) as image:
            tensor = runtime.transform(image.convert('RGB')).unsqueeze(0).cuda()
            score, mask = predict_single_image(runtime.model.model, runtime.model.decoder,
                                               runtime.model.predictor, tensor, runtime.model.feat_size)
            result = runtime.predict(image)
        expected = captured['mask']
        actual = mask.detach().cpu()
        assert torch.equal(expected, actual), name
        assert torch.equal(scope['score'], score), name
        assert result['score'] == float(scope['score'].cpu()), name
        pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(result['native_mask']))))
        expected_pixels = (expected[0, 0].numpy() * 255).astype('uint8')
        assert np.array_equal(expected_pixels, pixels), name
        folder = args.output_dir / path.stem
        folder.mkdir(exist_ok=True)
        Image.fromarray(expected_pixels).save(folder / 'authors_pred.png')
        (folder / 'web_pred.png').write_bytes(base64.b64decode(result['native_mask']))
        np.savez_compressed(folder / 'maps.npz', authors=expected.numpy(), web=actual.numpy())
        samples.append(dict(sample=name, score=result['score'], score_equal=True,
                            float_mask_equal=True, native_png_pixels_equal=True,
                            max_abs_error=float((expected - actual).abs().max()),
                            native_shape=result['native_mask_shape'], vram_gb=result['vram_gb']))
        print(json.dumps(samples[-1]), flush=True)
    report = dict(
        scope='Exact original single-image inference statements, same official cached HF CLIP weights. '
              'Not an independent PKL weight comparison or benchmark reproduction.',
        source='https://github.com/MaticFuc/AnomalyVFM/blob/main/predict_single_image.py',
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        model_id=MODEL_ID, checkpoint_snapshot=config_path.parent.name,
        torch=torch.__version__, device=torch.cuda.get_device_name(), samples=samples)
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
