import importlib.util
import sys
import os
import types

base = os.path.join(os.path.dirname(__file__), 'venv_tfjs_311', 'lib', 'python3.11', 'site-packages')

# If the project-specific venv with tensorflowjs isn't present, fall back
# to the current Python environment's site-packages so the converter can
# use an installed `tensorflowjs` package.
if not os.path.isdir(base):
    try:
        import sysconfig
        site_packages = sysconfig.get_paths().get('purelib')
    except Exception:
        site_packages = None
    if not site_packages:
        site_packages = next((p for p in sys.path if 'site-packages' in p), None)
    if site_packages:
        base = site_packages
    else:
        print('Cannot locate site-packages for tensorflowjs. Install tensorflowjs or create venv_tfjs_311.')
        sys.exit(1)

def load_module_from_file(fullname, filepath):
    spec = importlib.util.spec_from_file_location(fullname, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)
    return mod

os.makedirs('web_demo/tfjs_model', exist_ok=True)

# Prepare minimal package entries so importing keras_h5_conversion succeeds
if 'tensorflowjs' not in sys.modules:
    sys.modules['tensorflowjs'] = types.ModuleType('tensorflowjs')

# load version first (common imports it)
version_path = os.path.join(base, 'tensorflowjs', 'version.py')
version = load_module_from_file('tensorflowjs.version', version_path)
setattr(sys.modules['tensorflowjs'], 'version', version)

# load converters.common
common_path = os.path.join(base, 'tensorflowjs', 'converters', 'common.py')
common = load_module_from_file('tensorflowjs.converters.common', common_path)
if 'tensorflowjs.converters' not in sys.modules:
    sys.modules['tensorflowjs.converters'] = types.ModuleType('tensorflowjs.converters')
setattr(sys.modules['tensorflowjs.converters'], 'common', common)

# Provide a dummy tensorflowjs.write_weights module so the import in
# `keras_h5_conversion` succeeds (we implement writing ourselves below).
if 'tensorflowjs.write_weights' not in sys.modules:
    sys.modules['tensorflowjs.write_weights'] = types.ModuleType('tensorflowjs.write_weights')
setattr(sys.modules['tensorflowjs'], 'write_weights', sys.modules['tensorflowjs.write_weights'])

# Now load keras_h5_conversion (it will import the above stubs)
kh5_path = os.path.join(base, 'tensorflowjs', 'converters', 'keras_h5_conversion.py')
kh5 = load_module_from_file('tensorflowjs.converters.keras_h5_conversion', kh5_path)

print('Running conversion (custom writer)...')
model_json, groups = kh5.h5_merged_saved_model_to_tfjs_format('cats_dogs_model.h5')

OUT_DIR = 'web_demo/tfjs_model'
SHARD_BYTES = 4 * 1024 * 1024

def serialize_entry_data(arr):
    if arr.dtype == object:
        # serialize strings: 4-byte little-endian length + bytes
        out = bytearray()
        flat = arr.flatten().tolist()
        for s in flat:
            b = s.encode('utf-8') if isinstance(s, str) else s
            out += (len(b)).to_bytes(4, 'little')
            out += b
        return bytes(out)
    else:
        return arr.tobytes()

manifest = []
for gi, group in enumerate(groups):
    # collect bytes
    parts = []
    weights_info = []
    for entry in group:
        data = entry['data']
        parts.append(serialize_entry_data(data))
        dtype = 'string' if data.dtype == object else data.dtype.name
        weights_info.append({'name': entry['name'], 'shape': list(data.shape), 'dtype': dtype})

    total = sum(len(p) for p in parts)
    if total == 0:
        paths = []
    else:
        num_shards = max(1, (total + SHARD_BYTES - 1) // SHARD_BYTES)
        paths = []
        cur = 0
        buf = b''.join(parts)
        for si in range(num_shards):
            start = si * SHARD_BYTES
            end = min(start + SHARD_BYTES, len(buf))
            fname = f'group{gi+1}-shard{si+1}of{num_shards}.bin'
            with open(os.path.join(OUT_DIR, fname), 'wb') as f:
                f.write(buf[start:end])
            paths.append(fname)

    manifest.append({'paths': paths, 'weights': weights_info})

import json
with open(os.path.join(OUT_DIR, 'weights_manifest.json'), 'w') as f:
    json.dump(manifest, f)

# Build model.json in TFJS Layers format
model_json_out = {
    'format': 'layers-model',
    'generatedBy': f"keras v{model_json.get('keras_version', '')}",
    'convertedBy': 'custom-converter',
    'modelTopology': model_json.get('model_config', None),
    'weightsManifest': manifest
}
with open(os.path.join(OUT_DIR, 'model.json'), 'w') as f:
    json.dump(model_json_out, f)

print('Conversion complete: web_demo/tfjs_model')
