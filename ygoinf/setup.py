from setuptools import setup, find_packages

__version__ = "0.0.1"

INSTALL_REQUIRES = [
  "numpy==1.26.4",
  "optree",
  "fastapi",
  "uvicorn[standard]",
  "pydantic_settings",
]

EXTRAS_REQUIRE = {
  "jax": [
    "jax<=0.4.28",
    "jaxlib<=0.4.28",
    "flax",
  ],
  "tflite": [
    "tflite-runtime; platform_system != 'Windows'",
  ],
}
EXTRAS_REQUIRE["all"] = sorted({
    dep for deps in EXTRAS_REQUIRE.values() for dep in deps
})

setup(
    name="ygoinf",
    version=__version__,
    packages=find_packages(include='ygoinf*'),
    long_description="",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.10",
)
