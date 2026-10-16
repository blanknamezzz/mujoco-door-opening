"""Process-local correction for a malformed inherited CUDA allocator option."""
import os
import sys
import warnings


def configure_torch_environment():
    for name in ['PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF']:
        value = os.environ.get(name, '')
        tokens = value.split(',')
        if not any(token.strip().startswith('expandable_segments=') for token in tokens):
            continue
        corrected = []
        for token in tokens:
            token = token.strip()
            if token.startswith('expandable_segments='):
                # PyTorch expects ':', and its expandable allocator is not
                # supported by the Windows build used in this project.
                if sys.platform == 'win32':
                    continue
                token = token.replace('=', ':', 1)
            corrected.append(token)
        if corrected:
            os.environ[name] = ','.join(corrected)
        else:
            os.environ.pop(name, None)
        warnings.warn(f'Corrected malformed expandable_segments option in {name} for this process only.',
                      RuntimeWarning, stacklevel=2)


configure_torch_environment()
