"""Versioned data ingestion with explicit graph coverage and split provenance."""

import json
import tarfile
from pathlib import Path

import numpy as np

from common import checked_download, digest, read_json, write_json

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
SOURCES = read_json(Path(__file__).with_name("sources.json"))


def download(root="data/raw", target="all"):
    root = Path(root)
    for name, spec in SOURCES.items():
        if target == "all" or (target == "cifar10") == name.endswith(".gz"):
            checked_download(root / name, spec)


def exact_ids(values):
    values = np.asarray(values)
    if values.dtype.kind not in "iu" or np.any(values < 0):
        raise ValueError("Body IDs must be nonnegative integers, never floating point")
    if values.size and values.max() > np.iinfo(np.int64).max:
        raise ValueError("Body ID exceeds supported int64 range")
    return values.astype(np.int64)


def index_edges(ids, pre, post, counts):
    if not len(ids) or np.any(ids[1:] <= ids[:-1]):
        raise ValueError("Node IDs must be nonempty, unique and sorted")
    pre, post = exact_ids(pre), exact_ids(post)
    counts = np.asarray(counts)
    if not (len(pre) == len(post) == len(counts)):
        raise ValueError("Edge column lengths differ")
    if not np.all(np.isfinite(counts)) or np.any(counts < 1) or np.any(counts != np.floor(counts)):
        raise ValueError("Contact counts must be positive integers")
    a, b = np.searchsorted(ids, pre), np.searchsorted(ids, post)
    keep = (a < len(ids)) & (b < len(ids))
    keep &= ids[np.minimum(a, len(ids) - 1)] == pre
    keep &= ids[np.minimum(b, len(ids) - 1)] == post
    return a[keep], b[keep], counts[keep], keep


def save_graph(output, ids, pre, post, counts, sign, report):
    from scipy.sparse import csr_matrix
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    n = len(ids)
    # Matrix rows are destinations; columns are sources. Parallel rows sum contacts.
    matrix = csr_matrix((np.asarray(counts, dtype=np.float64), (post, pre)), shape=(n, n))
    matrix.sum_duplicates()
    matrix.sort_indices()
    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    values = matrix.data * sign[matrix.indices]
    values /= np.repeat(np.maximum(row_sum, 1), np.diff(matrix.indptr))
    if np.any(values == 0) or not np.isfinite(values).all():
        raise ValueError("Normalization must not remove or invalidate retained edges")
    report = dict(report, nodes=n, edges=matrix.nnz,
                  retained_contact_count=int(matrix.data.sum()),
                  duplicate_rows_aggregated=len(pre) - matrix.nnz,
                  isolated_neurons=int(np.count_nonzero((row_sum == 0) & (np.asarray(matrix.sum(axis=0)).ravel() == 0))),
                  normalization="Signed contact counts / total incoming contact count. No edge threshold.",
                  sign_policy="GABA and glutamate: -1; all other/missing transmitters: +1. Modeling assumption, not receptor physiology.")
    path = output / "graph.npz"
    with (output / "graph.npz.partial").open("wb") as f:
        np.savez_compressed(f, ids=ids, indptr=matrix.indptr.astype(np.int64),
                            indices=matrix.indices.astype(np.int64), values=values.astype(np.float32),
                            contacts=matrix.data.astype(np.int64), signs=np.asarray(sign, dtype=np.float32),
                            metadata=np.asarray(json.dumps(report)))
    (output / "graph.npz.partial").replace(path)
    report["graph_sha256"] = digest(path)
    write_json(output / "report.json", report)
    return report


def prepare_connectome(raw, output, verify=True):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc

    raw, output = Path(raw), Path(output)
    hashes = {}
    for name in ("annotations.feather", "neurotransmitters.feather", "edges.feather"):
        hashes[name] = digest(raw / name)
        if verify and hashes[name] != SOURCES[name]["sha256"]:
            raise ValueError(f"Unverified MaleCNS source: {name}")
    frame = feather.read_table(raw / "annotations.feather").to_pandas()
    nt = feather.read_table(raw / "neurotransmitters.feather").to_pandas()
    for required in ("bodyId", "superclass", "status"):
        if required not in frame:
            raise ValueError(f"Missing annotation column: {required}")
    ids_all = exact_ids(frame.bodyId.to_numpy())
    if len(np.unique(ids_all)) != len(ids_all) or nt.body.duplicated().any():
        raise ValueError("Duplicate biological IDs")
    keep_nodes = frame.superclass.notna() & frame.superclass.ne("") & frame.status.ne("Glia")
    nodes = frame.loc[keep_nodes].sort_values("bodyId").copy()
    ids = exact_ids(nodes.bodyId.to_numpy())
    nts = nodes.bodyId.map(nt.set_index("body").consensus_nt).fillna("unknown").astype(str).str.lower()
    signs = np.where(nts.isin(["gaba", "glutamate"]), -1.0, 1.0)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"body_id": ids, "superclass": nodes.superclass.to_numpy(),
                  "neurotransmitter": nts.to_numpy(), "modeled_sign": signs}).to_csv(output / "neurons.csv", index=False)
    excluded = frame.loc[~keep_nodes, ["bodyId", "superclass", "status"]]
    excluded.to_csv(output / "excluded_objects.csv", index=False)

    reader = ipc.open_file(pa.memory_map(str(raw / "edges.feather"), "r"))
    stats = {"source_edge_rows": 0, "retained_edge_rows": 0, "source_contact_count": 0,
             "retained_contact_count": 0, "retained_self_edges": 0, "retained_weight_one_edges": 0}

    def batches():
        for j in range(reader.num_record_batches):
            batch = reader.get_batch(j)
            cols = [batch.column(batch.schema.get_field_index(k)).to_numpy(zero_copy_only=False)
                    for k in ("body_pre", "body_post", "weight")]
            yield cols, index_edges(ids, *cols)

    print("Auditing every released connection …", flush=True)
    for cols, (pre, post, count, mask) in batches():
        stats["source_edge_rows"] += len(mask)
        stats["retained_edge_rows"] += len(pre)
        stats["source_contact_count"] += int(np.sum(cols[2], dtype=np.int64))
        stats["retained_contact_count"] += int(np.sum(count, dtype=np.int64))
        stats["retained_self_edges"] += int(np.count_nonzero(pre == post))
        stats["retained_weight_one_edges"] += int(np.count_nonzero(count == 1))
    if verify and (len(ids) != 166700 or stats["retained_edge_rows"] != 25582938):
        raise ValueError(f"Coverage changed: {len(ids)} nodes, {stats['retained_edge_rows']} edges. Audit before proceeding.")
    pre = np.empty(stats["retained_edge_rows"], dtype=np.int32)
    post = np.empty_like(pre)
    counts = np.empty(len(pre), dtype=np.int64)
    offset = 0
    for _, (a, b, c, _) in batches():
        end = offset + len(a)
        pre[offset:end], post[offset:end], counts[offset:end] = a, b, c
        offset = end
    stats["excluded_edge_rows"] = stats["source_edge_rows"] - stats["retained_edge_rows"]
    stats["excluded_contact_count"] = stats["source_contact_count"] - stats["retained_contact_count"]
    report = {
        "kind": "malecns_v1" if verify else "test_fixture", "source_verified": verify,
        "source_hashes": hashes, "source": "https://male-cns.janelia.org/download/",
        "license": "CC-BY-4.0", "source_annotation_rows": len(frame),
        "excluded_annotation_rows": len(excluded),
        "node_policy": "All assigned neuronal superclasses, including uncertain classes, except explicit Glia; no Traced-status filter.",
        "upstream_filter": "MaleCNS v1.0 minconf-0.5 release; not all segmentation fragments are annotated neurons.",
        "superclass_counts": nodes.superclass.value_counts().to_dict(),
        "transmitter_counts": nts.value_counts().to_dict(), "audit": stats,
    }
    return save_graph(output, ids, pre, post, counts, signs, report)


def save_images(output, images, labels, train_idx, val_idx, test_images, test_labels, report):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        "train_images": images[train_idx], "train_labels": labels[train_idx],
        "val_images": images[val_idx], "val_labels": labels[val_idx],
        "test_images": test_images, "test_labels": test_labels,
        "train_source_indices": train_idx, "val_source_indices": val_idx,
    }
    hashes = {}
    for name, array in arrays.items():
        path = output / (name + ".npy")
        with path.with_suffix(".partial").open("wb") as f:
            np.save(f, array, allow_pickle=False)
        path.with_suffix(".partial").replace(path)
        hashes[path.name] = digest(path)
    report = dict(report, classes=CLASSES, image_size=images.shape[-1], files=hashes,
                  train_examples=len(train_idx), val_examples=len(val_idx), test_examples=len(test_labels))
    write_json(output / "manifest.json", report)
    return report


def prepare_cifar(raw, output):
    path = Path(raw) / "cifar-10-binary.tar.gz"
    if digest(path) != SOURCES[path.name]["sha256"]:
        raise ValueError("CIFAR archive checksum mismatch")
    with tarfile.open(path, "r:gz") as archive:
        def read_batch(name):
            member = archive.getmember("cifar-10-batches-bin/" + name + ".bin")
            if not member.isfile() or member.size != 30730000:
                raise ValueError("Unexpected CIFAR member size/type")
            # Read named files directly; no extraction paths and no pickle execution.
            with archive.extractfile(member) as f:
                rows = np.frombuffer(f.read(), dtype=np.uint8).reshape(10000, 3073)
            if np.any(rows[:, 0] >= 10):
                raise ValueError("Invalid class in CIFAR archive")
            return rows[:, 1:].reshape(-1, 3, 32, 32).copy(), rows[:, 0].astype(np.int64)
        parts = [read_batch(f"data_batch_{i}") for i in range(1, 6)]
        images = np.concatenate([x for x, _ in parts])
        labels = np.concatenate([y for _, y in parts])
        test_images, test_labels = read_batch("test_batch")
    rng = np.random.default_rng(1729)
    training, validation = [], []
    for label in range(10):
        idx = np.flatnonzero(labels == label)
        if len(idx) != 5000:
            raise ValueError("Unexpected CIFAR class balance")
        rng.shuffle(idx)
        validation.extend(idx[:500])
        training.extend(idx[500:])
    return save_images(output, images, labels, np.sort(training), np.sort(validation), test_images, test_labels,
                       {"kind": "cifar10", "archive_sha256": digest(path), "split_seed": 1729,
                        "source": SOURCES[path.name]["url"], "normalization": "uint8 NCHW; training maps to [-1, 1]",
                        "split_policy": "500 held-out validation images per training class; official 10000 test images untouched"})


def prepare(root="data", target="all"):
    root = Path(root)
    if target in ("all", "malecns"):
        out = root / "processed/malecns"
        if (out / "report.json").exists():
            if digest(out / "graph.npz") != read_json(out / "report.json")["graph_sha256"]:
                raise ValueError("Existing processed graph is corrupt; use a new --root")
            print("Verified existing processed connectome", flush=True)
        else:
            print(json.dumps(prepare_connectome(root / "raw", out), indent=2), flush=True)
    if target in ("all", "cifar10"):
        out = root / "processed/cifar10"
        if (out / "manifest.json").exists():
            for name, expected in read_json(out / "manifest.json")["files"].items():
                if digest(out / name) != expected:
                    raise ValueError(f"Corrupt prepared image data: {name}")
            print("Verified existing prepared CIFAR-10", flush=True)
        else:
            print(json.dumps(prepare_cifar(root / "raw", out), indent=2), flush=True)


def fixture(root):
    """Deliberately synthetic mini-network, NEVER a substitute for the full experiment."""
    root = Path(root)
    if root.exists():
        raise ValueError(f"Fixture directory already exists: {root}")
    rng = np.random.default_rng(100)
    n = 64
    src = np.repeat(np.arange(n), 4)
    dst = np.concatenate([(np.arange(4) + i + 1) % n for i in range(n)])
    save_graph(root / "graph", np.arange(n), src, dst, np.ones(len(src)),
               np.where(np.arange(n) % 4 == 0, -1, 1), {"kind": "synthetic_fixture", "source_verified": False})
    images = rng.integers(0, 256, (120, 3, 8, 8), dtype=np.uint8)
    labels = np.arange(120) % 10
    save_images(root / "images", images[:100], labels[:100], np.arange(80), np.arange(80, 100),
                images[100:], labels[100:], {"kind": "synthetic_fixture"})
    config = {
        "graph": str((root / "graph/graph.npz").resolve()), "dataset": str((root / "images").resolve()),
        "seed": 42, "threads": 1, "image_size": 8, "width": 16, "recurrent_steps": 3, "leak": 0.5,
        "diffusion_steps": 32, "batch_size": 4, "train_steps": 8, "learning_rate": 0.001,
        "weight_decay": 0.0, "gradient_clip": 1.0, "ema_decay": 0.9,
        "checkpoint_every": 4, "validate_every": 4, "validation_batches": 2, "log_every": 2,
        "allow_synthetic": True, "graph_control": "real",
    }
    write_json(root / "config.json", config)
    return config


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download, verify and prepare full MaleCNS + CIFAR-10.")
    parser.add_argument("--root", default="data")
    parser.add_argument("--target", choices=["all", "malecns", "cifar10"], default="all")
    parser.add_argument("--offline", action="store_true", help="Only preprocess already downloaded files")
    parser.add_argument("--fixture", action="store_true", help="Create tiny synthetic test data, NOT fly data")
    parser.add_argument("--weights-url", help="Optional HTTPS URL of published model.pt (not yet hosted by this project)")
    parser.add_argument("--weights-sha256", help="Required trusted SHA256 for --weights-url")
    args = parser.parse_args()
    if bool(args.weights_url) != bool(args.weights_sha256) or (args.weights_url and (args.offline or args.fixture)):
        parser.error("Use --weights-url and --weights-sha256 together, without --offline/--fixture")
    if args.fixture:
        fixture(args.root)
    else:
        if not args.offline:
            download(Path(args.root) / "raw", args.target)
        prepare(args.root, args.target)
        if args.weights_url:
            checked_download(Path(args.root) / "weights/model.pt", {"url": args.weights_url, "sha256": args.weights_sha256})
