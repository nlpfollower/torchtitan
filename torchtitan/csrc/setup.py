from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension
import os

# Add C++17 standard
extra_compile_args = ['-std=c++17']
if os.name == 'nt':
    extra_compile_args = ['/std:c++17']

# Common include directories
base_include_dirs = [
    '/usr/include',
    '/usr/include/hiredis'
]

# Get Gloo paths from environment
gloo_include = os.environ.get('GLOO_INCLUDE_DIR')
gloo_lib = os.environ.get('GLOO_LIB_DIR')

# Initialize all extension configurations
gloo_include_dirs = base_include_dirs.copy()
gloo_library_dirs = []
gloo_libraries = ['gloo', 'hiredis']

if gloo_include:
    gloo_include_dirs.append(gloo_include)
    gloo_include_dirs.append(os.path.join(gloo_include, 'build'))
if gloo_lib:
    gloo_library_dirs.append(gloo_lib)

# Define extensions
tensor_preloader_ext = cpp_extension.CppExtension(
    name='tensor_preloader',
    sources=[
        'tensor_common.cpp',
        'thread_pool.cpp',
        'memory_mapped_file.cpp',
        'shared_memory.cpp',
        'gloo_file_broadcast.cpp',
        'tensor_preloader.cpp'
    ],
    include_dirs=gloo_include_dirs,
    library_dirs=gloo_library_dirs,
    libraries=gloo_libraries,
    extra_compile_args=extra_compile_args,
)

# For the fast_tensor_loader, we don't need Gloo dependencies
fast_tensor_loader_ext = cpp_extension.CppExtension(
    name='fast_tensor_loader',
    sources=[
        'tensor_common.cpp',
        'shared_memory.cpp',
        'fast_tensor_loader.cpp'
    ],
    include_dirs=base_include_dirs,
    extra_compile_args=extra_compile_args,
)

setup(
    name='tensor_loading_extensions',
    version='0.1.0',
    description='Fast tensor loading and preloading extensions with distributed support',
    author='Your Name',
    author_email='your.email@example.com',
    ext_modules=[
        tensor_preloader_ext,
        fast_tensor_loader_ext
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension.with_options(no_python_abi_suffix=True)
    },
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.9.0',
        'redis>=3.5.0',
    ],
)