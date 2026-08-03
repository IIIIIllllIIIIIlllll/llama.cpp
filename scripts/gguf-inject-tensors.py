#!/usr/bin/env python3
# inject token_embd.weight + output.weight from a donor GGUF into a target GGUF
# (pure python, appends tensors at the end of the data section)
import struct, sys, shutil

BLOCK = {  # ggml type -> (block_size, type_size)
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 9: (32, 40), 10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 240), 16: (32, 34+4), 30: (1, 4),
}

def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode()

VAL_SIZE = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}

def skip_value(f, vtype):
    if vtype == 8:
        n = struct.unpack('<Q', f.read(8))[0]; f.seek(n, 1)
    elif vtype == 9:
        etype = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        if etype == 8:
            for _ in range(n): skip_value(f, 8)
        else:
            f.seek(VAL_SIZE[etype] * n, 1)
    else:
        f.seek(VAL_SIZE[vtype], 1)

def parse(path, want_tensors=()):
    """returns dict with section offsets, alignment, and tensor infos"""
    with open(path, 'rb') as f:
        assert f.read(4) == b'GGUF'
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors, n_kv = struct.unpack('<QQ', f.read(16))
        alignment = 32
        for _ in range(n_kv):
            key = read_str(f)
            vtype = struct.unpack('<I', f.read(4))[0]
            if key == 'general.alignment':
                alignment = struct.unpack('<I', f.read(4))[0]
            else:
                skip_value(f, vtype)
        kv_end = f.tell()
        tensors = {}
        first_off = None
        for _ in range(n_tensors):
            pos = f.tell()
            name = read_str(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = struct.unpack(f'<{n_dims}Q', f.read(8 * n_dims))
            ttype = struct.unpack('<I', f.read(4))[0]
            off = struct.unpack('<Q', f.read(8))[0]
            if first_off is None: first_off = pos
            if name in want_tensors or not want_tensors:
                tensors[name] = (dims, ttype, off)
        infos_end = f.tell()
        import os
        data_start = (infos_end + alignment - 1) // alignment * alignment
        filesize = os.path.getsize(path)
        return dict(version=version, n_kv=n_kv, n_tensors=n_tensors, alignment=alignment,
                    kv_end=kv_end, infos_start=first_off, infos_end=infos_end,
                    data_start=data_start, filesize=filesize, tensors=tensors)

def tensor_size(dims, ttype):
    n = 1
    for d in dims: n *= d
    bs, ts = BLOCK[ttype]
    assert n % bs == 0, f'{n} not multiple of {bs}'
    return n // bs * ts

def main(dst_path, donor_path, out_path, names):
    dst = parse(dst_path)
    donor = parse(donor_path, want_tensors=names)
    for n in names:
        assert n in donor['tensors'], f'{n} missing in donor'
    align = dst['alignment']
    print(f'dst: {dst["n_tensors"]} tensors, align {align}, data_start {dst["data_start"]}, size {dst["filesize"]}')

    with open(dst_path, 'rb') as f:
        f.seek(24)  # skip magic+version+n_tensors+n_kv (rewritten)
        kv_blob = f.read(dst['kv_end'] - 24)
        f.seek(dst['infos_start'])
        infos_blob = f.read(dst['infos_end'] - dst['infos_start'])

    # new tensor infos
    old_data_len = dst['filesize'] - dst['data_start']
    new_infos = b''
    cursor = (old_data_len + align - 1) // align * align
    appended = []
    with open(donor_path, 'rb') as df:
        for n in names:
            dims, ttype, off = donor['tensors'][n]
            size = tensor_size(dims, ttype)
            nb = n.encode()
            info = struct.pack('<Q', len(nb)) + nb
            info += struct.pack('<I', len(dims))
            info += struct.pack(f'<{len(dims)}Q', *dims)
            info += struct.pack('<I', ttype)
            info += struct.pack('<Q', cursor)
            new_infos += info
            df.seek(donor['data_start'] + off)
            data = df.read(size)
            assert len(data) == size
            appended.append((n, cursor, data))
            cursor = (cursor + size + align - 1) // align * align
            print(f'  + {n}: dims={dims} type={ttype} size={size} new_off={appended[-1][1]}')

    with open(dst_path, 'rb') as src, open(out_path, 'wb') as out:
        out.write(b'GGUF')
        out.write(struct.pack('<I', dst['version']))
        out.write(struct.pack('<Q', dst['n_tensors'] + len(names)))
        out.write(struct.pack('<Q', dst['n_kv']))
        out.write(kv_blob)
        out.write(infos_blob)
        out.write(new_infos)
        # pad to alignment
        pos = out.tell()
        pad = (pos + align - 1) // align * align - pos
        out.write(b'\x00' * pad)
        # copy old data section
        src.seek(dst['data_start'])
        shutil.copyfileobj(src, out, 1 << 24)
        # append new tensors (offsets relative to data start; pad between)
        cur = old_data_len
        for n, off, data in appended:
            assert cur <= off
            out.write(b'\x00' * (off - cur))
            out.write(data)
            cur = off + len(data)
    print(f'wrote {out_path}')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         ['token_embd.weight', 'output.weight'])
