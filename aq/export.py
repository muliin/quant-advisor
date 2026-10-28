"""模型量化导出：按 NF4 量化并保存为可重新加载的产物。

三条路线：

- :func:`export_bnb_4bit`：bnb NF4 量化 + ``save_pretrained``，默认路线
- :func:`export_skip_layers`：整体 NF4，敏感层保留 bf16 的混合精度
- 伪量化路线：按 4bit 计算、以 bf16 保存，文件大小不变，仅用于验证量化效果

Note:
    bnb 在 transformers 5.19 下不量化 OLMoE 的融合专家张量，MoE 模型走
    ``export_bnb_4bit`` 会得到 12.5 GiB，压缩无效；此类模型只能走伪量化路线，
    或改用自定义存储格式。

Warning:
    vLLM 不支持逐专家位宽，因此可部署的混合精度只能表达为
    「NF4 + 若干整层 bf16」。

用法
  python -m aq.export --model Qwen/Qwen2.5-1.5B-Instruct --out out/qwen4bit
  python -m aq.export --model Qwen/Qwen2.5-1.5B-Instruct --out out/qwen4bit_skip \\
      --skip model.layers.27.mlp.down_proj,model.layers.26.mlp.down_proj
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

GIB = 1024 ** 3

__all__ = ["export_bnb_4bit", "export_skip_layers", "dir_size_gib",
           "verify_loadable", "export_report"]


def dir_size_gib(path: str | Path) -> float:
    """递归统计目录下所有文件的总体积。

    Args:
        path: 目录路径。

    Returns:
        总体积，单位 GiB；目录不存在时返回 0.0。
    """
    p = Path(path)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / GIB


def export_bnb_4bit(model_ref: str, out_dir: str, skip_modules: list[str] | None = None,
                    double_quant: bool = True, progress: bool = True,
                    bf16_gib: float | None = None) -> dict:
    """按 bnb NF4 量化模型并保存产物，可用 ``load_in_4bit=True`` 重新加载。

    参数与返回值中的大小一律以 GiB 计。

    Args:
        model_ref: Hugging Face 模型名或本地路径。
        out_dir: 输出目录；已存在时整个目录被删除后重建（不可逆）。
        skip_modules: 不参与量化的模块名列表，这些模块保留 bf16。名称不带
            ``.weight`` 后缀，例如 ``model.layers.27.mlp.down_proj``；
            该参数即混合精度部署的入口。
        double_quant: 是否启用 bnb 双重量化，即对量化常数再量化以进一步压缩。
        progress: 是否打印加载、量化、保存三阶段的进度。
        bf16_gib: 原始 bf16 权重的文件大小，单位 GiB。仅当给定时才返回压缩倍数。

    Returns:
        结果字典，键包含：``ok`` / ``out_dir`` / ``size_gib``（GiB）/
        ``params_packed``（打包后的元素数，非原始参数量）/ ``compression``（倍数）/
        ``gpu_gib``（量化时的显存占用，GiB）/ ``load_seconds``（秒）/
        ``skip_modules`` / ``double_quant``。

    Note:
        bnb 的 4bit 权重在显存中以打包形式存储，``numel()`` 返回打包后的
        uint8 元素数，与原始参数量不可直接比较。

    Warning:
        ``out_dir`` 已存在时会被 ``shutil.rmtree`` 删除，不可恢复。
    """
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=double_quant,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    kw = {"quantization_config": cfg, "device_map": {"": 0}}
    if skip_modules:
        kw["modules_to_not_convert"] = skip_modules

    t0 = time.perf_counter()
    if progress:
        print(f"  加载并量化 {model_ref} ...")
    tok = AutoTokenizer.from_pretrained(model_ref)
    model = AutoModelForCausalLM.from_pretrained(model_ref, **kw)
    load_s = time.perf_counter() - t0
    used = torch.cuda.memory_allocated() / GIB

    if progress:
        print(f"  量化后显存 {used:.3f} GiB，保存中 ...")
    model.save_pretrained(out)
    tok.save_pretrained(out)

    size = dir_size_gib(out)
    # NOTE: bnb 的 numel() 返回的是打包后的存储量（uint8 元素数），
    # 不等于原始参数量，不能用来计算 bits/param。
    n_par = sum(p.numel() for p in model.parameters())
    ratio = (bf16_gib / size) if bf16_gib else None

    back_bf16 = sum(
        1 for m in model.modules() if type(m).__name__ == "Linear"
        and not type(m).__name__.endswith("4bit")
    )

    del model
    torch.cuda.empty_cache()
    return {"ok": True, "out_dir": str(out), "size_gib": round(size, 4),
            "params_packed": n_par, "bf16_gib": bf16_gib,
            "compression": (round(ratio, 2) if ratio else None),
            "gpu_gib": round(used, 3), "load_seconds": round(load_s, 1),
            "skip_modules": skip_modules or [],
            "double_quant": double_quant}


def export_skip_layers(model_ref: str, out_dir: str,
                       sensitive: list[str], progress: bool = True) -> dict:
    """整体 NF4 量化，仅把 ``sensitive`` 中的层保留 bf16，得到可部署的混合精度。

    等价于以 ``skip_modules=sensitive`` 调用 :func:`export_bnb_4bit`，
    ``double_quant`` 等参数沿用其默认值。

    Args:
        model_ref: Hugging Face 模型名或本地路径。
        out_dir: 输出目录；已存在时整个目录被删除后重建（不可逆）。
        sensitive: 保留 bf16 的模块名列表，名称不带 ``.weight`` 后缀，
            例如 ``["model.layers.27.mlp.down_proj"]``。
        progress: 是否打印加载、量化、保存三阶段的进度。

    Returns:
        与 :func:`export_bnb_4bit` 结构相同的结果字典。
    """
    return export_bnb_4bit(model_ref, out_dir, skip_modules=sensitive,
                           progress=progress)


def verify_loadable(out_dir: str, progress: bool = True) -> dict:
    """重新加载导出的产物并执行一次前向，确认可加载且输出有限。

    Args:
        out_dir: :func:`export_bnb_4bit` 生成的目录。目录中须保留
            ``quantization_config``，否则模型会按原始精度加载。
        progress: 是否打印加载耗时与显存占用。

    Returns:
        结果字典：``ok`` 表示 logits 是否全为有限值，``load_gib`` 为加载后的
        显存占用（GiB），``seconds`` 为加载加前向的总耗时（秒）。

    Note:
        bnb 的 4bit 反量化在 CUDA 上完成，本函数要求可用的 GPU。
    """
    import math

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(out_dir)
    model = AutoModelForCausalLM.from_pretrained(out_dir).eval()
    if not next(model.parameters()).is_cuda:
        model = model.to("cuda")
    used = torch.cuda.memory_allocated() / GIB

    ids = tok("The quick brown fox jumps over the lazy dog. " * 8,
              return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        logits = model(ids).logits
    ok = torch.isfinite(logits).all().item()
    dt = time.perf_counter() - t0
    del model
    torch.cuda.empty_cache()
    if progress:
        print(f"  重新加载成功（{dt:.1f}s，{used:.3f} GiB），logits 有限={ok}")
    return {"ok": bool(ok), "load_gib": round(used, 3),
            "seconds": round(dt, 1)}


def export_report(res: dict, verify: dict | None = None) -> str:
    L = ["## 导出结果", "",
         "| 项 | 值 |", "|---|---|",
         f"| 输出目录 | `{res['out_dir']}` |",
         f"| **文件大小** | **{res['size_gib']:.3f} GiB** |",
         (f"| bf16 原始大小 | {res['bf16_gib']:.3f} GiB |\n"
          f"| **压缩倍数** | **{res['compression']:.2f}×** |")
         if res.get("bf16_gib") else
         f"| 量化后参数量（打包） | {res['params_packed']:,} |",
         f"| 量化时显存 | {res['gpu_gib']:.3f} GiB |",
         f"| double quant | {'是' if res['double_quant'] else '否'} |",
         f"| 留 bf16 的模块 | {len(res['skip_modules'])} 个 |", ""]
    if res["skip_modules"]:
        L += ["保留 bf16 的模块：", ""]
        L += [f"- `{m}`" for m in res["skip_modules"]]
        L += [""]
    if verify:
        L += ["### 可加载性验证", "",
              f"- 重新加载占用 **{verify['load_gib']:.3f} GiB**",
              f"- 前向输出有限：**{'✅ 是' if verify['ok'] else '❌ 否'}**",
              f"- 加载耗时 {verify['seconds']:.1f}s", ""]
    L += ["> ⚠️ **bnb 的 4bit 只提速显存，不提速推理**"
          "（实测带宽利用 9.9%，K013）。",
          "> 想提速要换推理后端（vLLM，K014）。"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="量化并导出模型")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", default="", help="逗号分隔的模块名（留 bf16）")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    skip = [x.strip() for x in a.skip.split(",") if x.strip()]
    res = export_skip_layers(a.model, a.out, skip) if skip \
        else export_bnb_4bit(a.model, a.out)
    ver = None if a.no_verify else verify_loadable(a.out)
    rep = export_report(res, ver)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "EXPORT_REPORT.md").write_text(rep, encoding="utf-8")
    (Path(a.out) / "export_meta.json").write_text(
        json.dumps({"res": res, "verify": ver}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()
