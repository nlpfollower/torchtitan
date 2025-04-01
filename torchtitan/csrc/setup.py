from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension
import os

# Add C++17 standard which is needed for some advanced LibTorch features
extra_compile_args = ['-std=c++17']
# For Windows
if os.name == 'nt':
    extra_compile_args = ['/std:c++17']

setup(
    name='tensor_loading_extensions',
    version='0.1.0',
    description='Fast tensor loading and preloading extensions using PyTorch C++ API',
    author='Your Name',
    author_email='your.email@example.com',
    ext_modules=[
        # Fast tensor loader
        cpp_extension.CppExtension(
            name='fast_tensor_loader',
            sources=['fast_tensor_loader.cpp'],
            extra_compile_args=extra_compile_args,
        ),
        # Tensor preloader extension
        cpp_extension.CppExtension(
            name='tensor_preloader',
            sources=['tensor_preloader.cpp'],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension.with_options(no_python_abi_suffix=True)
    },
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.9.0',
    ],
)