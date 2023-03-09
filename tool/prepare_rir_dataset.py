import argparse
import os
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rir_room_dir", required=True, type=str, help="room dir")
    parser.add_argument("--out_dir", required=True, type=str, help="output dir")
    parser.add_argument("--lmdb", required=False, default=False)
    args = parser.parse_args()

    assert os.path.exists(os.path.join(args.rir_room_dir, "rir_list"))
    rir_list = os.path.join(args.rir_room_dir, "rir_list")
    wav_scp = {}
    cur_workpath = os.getcwd()
    with open(rir_list, 'r') as f:
        for line in f:
            _, rir_id, _, _, path = line.strip().split()
            os.chdir(os.path.join(args.rir_room_dir, "../.."))
            assert os.path.exists(path)
            abs_path = os.path.abspath(path)
            wav_scp[rir_id] = abs_path

    os.chdir(cur_workpath)
    with open(os.path.join(args.out_dir, "wav.scp"), 'w') as f:
        for idx, (key, value) in enumerate(tqdm(wav_scp.items())):
            f.write(key + ' ' + value + '\n')

    if args.lmdb:
        import lmdb, math, pickle
        keys = []
        out_lmdb = os.path.join(args.out_dir, "rir.lmdb")
        db = lmdb.open(out_lmdb, map_size=int(math.pow(1024, 4)))  # 1TB
        txn = db.begin(write=True)
        for i, (key, value) in enumerate(tqdm(wav_scp.items())):
            keys.append(key)
            with open(value, 'rb') as fin:
                data = fin.read()
            txn.put(key.encode(), data)
            if i % 100 == 0:
                txn.commit()
                txn = db.begin(write=True)
        txn.commit()
        with db.begin(write=True) as txn:
            txn.put(b'__keys__', pickle.dumps(keys))
        db.sync()
        db.close()
            