
import numpy as np
import pandas as pd


try:
    from sdv.single_table import CTGANSynthesizer as Synth
    synth_name = "CTGANSynthesizer"
except Exception:
    from sdv.single_table import GaussianCopulaSynthesizer as Synth
    synth_name = "GaussianCopulaSynthesizer"


def make_mock_data(n_rows=50_000, n_merchants=20_000, seed=7):
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, n_merchants + 1)
    probs = 1 / ranks
    probs = probs / probs.sum()
    merchant_ids = [f"m_{i:06d}" for i in range(n_merchants)]
    merchants = rng.choice(merchant_ids, size=n_rows, p=probs)

    countries = rng.choice(["US","DE","FR","ES","IT","GB","NL","SE","NO"], size=n_rows)
    amount = np.round(rng.gamma(shape=2.0, scale=25.0, size=n_rows), 2)

    # Inject some nulls
    mask = rng.random(n_rows) < 0.01
    merchants = np.where(mask, None, merchants)

    df = pd.DataFrame({"merchant_id": merchants, "country": countries, "amount": amount})
    return df


def main():
    df = make_mock_data()
    print("Real data cardinality (merchant_id):", df["merchant_id"].nunique())

    base = Synth(epochs=40) if synth_name == "CTGANSynthesizer" else Synth()
    adapter = HighCardinalityAdapter(
        base_synthesizer=base,
        cardinality_threshold=500,
        top_k=200,
        n_buckets=512,
        seed=42,
        columns=["merchant_id"]
    )

    print(f"Fitting {synth_name} via HighCardinalityAdapter...")
    adapter.fit(df)

    print("Sampling synthetic data...")
    synth = adapter.sample(num_rows=10_000)
    print(synth.head())

    synth.to_csv("synthetic_sample.csv", index=False)
    print("Saved synthetic_sample.csv")


if __name__ == "__main__":
    main()
