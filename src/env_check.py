"""启动前的环境检查：把 ModuleNotFoundError 换成能直接照做的提示。"""
import sys
from pathlib import Path

# 模块名 -> pip 包名
REQUIRED = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
    "joblib": "joblib",
}


def check_environment() -> None:
    missing = []
    for module, package in REQUIRED.items():
        try:
            __import__(module)
        except ModuleNotFoundError:
            missing.append(package)
    if not missing:
        return

    root = Path(__file__).resolve().parent.parent
    venv_py = root / ".venv" / "Scripts" / "python.exe"
    venv_ok = venv_py.exists()

    lines = [
        "",
        "=" * 68,
        "  依赖缺失，无法运行",
        "=" * 68,
        f"  当前使用的解释器：{sys.executable}",
        f"  缺少的包：{', '.join(missing)}",
        "",
        "  原因：VS Code / 终端当前选的是别的 Python（例如 conda 的 base），",
        "        而本项目的依赖装在项目自带的虚拟环境 .venv 里。",
        "",
        "  ── 方案 A（推荐，零安装）─────────────────────────────",
    ]
    if venv_ok:
        lines += [
            f"  直接用项目虚拟环境运行：",
            f'    & "{venv_py}" "{root / "run_all.py"}"',
            "  或在项目目录下双击 run.bat。",
            "",
            "  VS Code 里切换解释器：",
            "    Ctrl+Shift+P -> Python: Select Interpreter",
            f'    -> 选择 "{venv_py}"',
        ]
    else:
        lines += [
            "  未找到 .venv，请先创建：",
            f'    cd "{root}"',
            "    python -m venv .venv",
            "    .venv\\Scripts\\pip install -r requirements.txt",
        ]
    lines += [
        "",
        "  ── 方案 B（装到当前环境）─────────────────────────────",
        f'    "{sys.executable}" -m pip install -r "{root / "requirements.txt"}"',
        "=" * 68,
        "",
    ]
    sys.stderr.write("\n".join(lines))
    raise SystemExit(1)
