import argparse
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

FP_SIZE = 1024
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_GROUPS = ("AVILMFYW", "KRH", "DE", "FWY")
CHARGES = {"D": -1, "E": -1, "R": 1, "K": 1, "H": 0.1}
HYDROPATHY = {
    "A": 1.8, "I": 4.5, "L": 3.8, "M": 1.9, "F": 2.8, "W": -0.9,
    "V": 4.2, "C": 2.5, "G": -0.4, "P": -1.6, "S": -0.8,
    "T": -0.7, "Y": -1.3, "N": -3.5, "Q": -3.5, "D": -3.5,
    "E": -3.5, "K": -3.9, "R": -4.5, "H": -3.2,
}
METADATA = {
    "Sequence", "Smiles", "ECNumber", "Organism", "Substrate",
    "type", "Test", "Name", "source", "Value",
}
MORGAN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=FP_SIZE, includeChirality=True
)


def protein_to_vector(sequence: str) -> np.ndarray:
    sequence = re.sub(r"[UZOB]", "X", sequence.upper())
    counts = Counter(sequence)
    length = len(sequence)
    ratio = 1 / length if length else 0
    features = [
        length / 500,
        len(counts) / 20,
        sum(counts[aa] * charge for aa, charge in CHARGES.items()) * ratio,
        sum(counts[aa] * value for aa, value in HYDROPATHY.items()) * ratio,
    ]
    features.extend(counts[aa] * ratio for aa in AMINO_ACIDS)
    features.extend(sum(counts[aa] for aa in group) * ratio for group in AMINO_GROUPS)
    return np.asarray(features, dtype=np.float32)


@lru_cache(maxsize=4096)
def smiles_to_vector(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    fingerprint = np.empty(FP_SIZE, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(MORGAN.GetFingerprint(mol), fingerprint)
    rotatable = Lipinski.NumRotatableBonds(mol)
    heavy_atoms = Lipinski.HeavyAtomCount(mol)
    molecular = np.asarray([
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
        Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol), rotatable,
        heavy_atoms, Lipinski.RingCount(mol), Lipinski.FractionCSP3(mol),
        Lipinski.NumAromaticRings(mol), rotatable / (heavy_atoms + 1e-6),
    ], dtype=np.float32)
    return np.concatenate((fingerprint, molecular))


@lru_cache(maxsize=4096)
def cached_protein_vector(sequence: str) -> np.ndarray:
    return protein_to_vector(sequence)


def load_protein_embeddings(names: pd.Series, h5_path: Path) -> np.ndarray:
    if names.isna().any():
        raise ValueError("Name Have Nan")

    embeddings = None
    cache = {}
    with h5py.File(h5_path, "r") as h5:
        for row, name in enumerate(tqdm(names, desc="Enzyme")):
            key = str(name)
            if key not in cache:
                if key not in h5:
                    raise KeyError(f"H5 dont have Name={key!r}（CSV  {row + 2} row）")
                values = np.asarray(h5[key][()], dtype=np.float32)
                if values.ndim == 2 and values.shape[0] > 0:
                    vector = values.mean(axis=0)
                elif values.ndim == 1:
                    vector = values
                else:
                    raise ValueError(f"H5  {key!r} Shape {values.shape}")
                if not vector.size or not np.isfinite(vector).all():
                    raise ValueError(f"H5  {key!r} HaveNan")
                cache[key] = vector

            vector = cache[key]
            if embeddings is None:
                embeddings = np.empty((len(names), len(vector)), dtype=np.float32)
            if len(vector) != embeddings.shape[1]:
                raise ValueError(f"H5 {key!r} dim is {len(vector)}，expectation {embeddings.shape[1]}")
            embeddings[row] = vector
    return embeddings


def build_features(data: pd.DataFrame, protein_h5: Path) -> np.ndarray:
    required = {"Name", "Smiles", "Sequence", "Value"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"CSV missing : {', '.join(sorted(missing))}")
    if data.empty:
        raise ValueError("CSV dont have sample")

    n_molecular = FP_SIZE + 11
    chemical_and_sequence = np.empty((len(data), n_molecular + 28), dtype=np.float32)
    records = zip(data["Smiles"], data["Sequence"])
    for row, (smiles, sequence) in enumerate(tqdm(records, total=len(data), desc="Phy")):
        if not isinstance(smiles, str) or not isinstance(sequence, str):
            raise ValueError(f"CSV  {row + 2} row is NAN")
        try:
            chemical_and_sequence[row, :n_molecular] = smiles_to_vector(smiles)
            chemical_and_sequence[row, n_molecular:] = cached_protein_vector(sequence)
        except ValueError as exc:
            raise ValueError(f"CSV  {row + 2} row: {exc}") from exc

    embeddings = load_protein_embeddings(data["Name"], protein_h5)
    parts = []
    extra_columns = [column for column in data if column not in METADATA]
    if extra_columns:
        parts.append(data[extra_columns].to_numpy(dtype=np.float32))
    parts.extend((chemical_and_sequence, embeddings))
    features = np.concatenate(parts, axis=1)
    if not np.isfinite(features).all():
        raise ValueError("NaN ，please check")
    return features


def evaluate(data: pd.DataFrame, features: np.ndarray, repeats: int = 5) -> pd.DataFrame:
    if repeats < 1:
        raise ValueError("repeats must > 0")
    values = pd.to_numeric(data["Value"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Value must > 0")
    labels = np.log10(values)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2,  shuffle=True
    )
    if len(y_train) < 2 or len(y_test) < 2:
        raise ValueError("Sample is small")
    print(f"Train: {len(y_train)}；Test: {len(y_test)}；Dim: {features.shape[1]}")

    results = []
    for run in range(repeats):
        model = ExtraTreesRegressor(n_estimators=100, n_jobs=-1)
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        mse = mean_squared_error(y_test, predicted)
        constant = np.all(y_test == y_test[0]) or np.all(predicted == predicted[0])
        results.append({
            "runs": run + 1,
            "R2": r2_score(y_test, predicted),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "MAE": mean_absolute_error(y_test, predicted),
            "PCC": np.nan if constant else float(pearsonr(y_test, predicted).statistic),
            "SCC": np.nan if constant else float(spearmanr(y_test, predicted).statistic),
        })

    report = pd.DataFrame(results)
    print("\nResult:\n", report.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nAvg Result:\n", report.drop(columns="runs").mean().to_string(float_format=lambda value: f"{value:.4f}"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="MVKcat-ET")
    parser.add_argument("csv", nargs="?", type=Path, default="./Data/XXX")
    parser.add_argument("--protein-h5", type=Path,
                        default="./Data/XXX")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    data = pd.read_csv(args.csv)
    features = build_features(data, args.protein_h5)
    evaluate(data, features, repeats=args.repeats)


if __name__ == "__main__":
    main()
