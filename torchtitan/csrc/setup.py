from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension
import os

# Add C++17 standard which is needed for some advanced LibTorch features
extra_compile_args = ['-std=c++17']
# For Windows
if os.name == 'nt':
    extra_compile_args = ['/std:c++17']

# Add Gloo dependencies
include_dirs = []
library_dirs = []
libraries = []

# Check for Gloo installation paths from environment variables
gloo_include = os.environ.get('GLOO_INCLUDE_DIR')
gloo_lib = os.environ.get('GLOO_LIB_DIR')

if gloo_include:
    include_dirs.append(gloo_include)
if gloo_lib:
    library_dirs.append(gloo_lib)

# Add Gloo and its dependencies
libraries.extend(['gloo', 'hiredis'])

setup(
    name='tensor_loading_extensions',
    version='0.1.0',
    description='Fast tensor loading and preloading extensions with distributed support',
    author='Your Name',
    author_email='your.email@example.com',
    ext_modules=[
        # Tensor preloader extension with Gloo support
        cpp_extension.CppExtension(
            name='tensor_preloader',
            sources=[
                'tensor_common.cpp',
                'thread_pool.cpp',
                'memory_mapped_file.cpp',
                'shared_memory.cpp',
                'gloo_file_broadcaster.cpp',  # Add the new source file
                'tensor_preloader.cpp'
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args,
        ),
        # Distributed tensor loader extension - simplified without Gloo direct dependency
        cpp_extension.CppExtension(
            name='distributed_tensor_loader',
            sources=[
                'tensor_common.cpp',
                'memory_mapped_file.cpp',
                'distributed_tensor_loader.cpp'
            ],
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
        'redis>=3.5.0',  # Add Redis client for Python
    ],
)