from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "tempo",
    "loudness",
    "liveness",
]


@dataclass(frozen=True)
class AgentMode:
    name: str
    description: str
    target: Dict[str, float]


MODES: Dict[str, AgentMode] = {
    "focus": AgentMode(
        name="Focus / Study",
        description="Calm, acoustic and low-energy music for concentration.",
        target={
            "danceability": 0.35,
            "energy": 0.30,
            "valence": 0.48,
            "acousticness": 0.75,
            "instrumentalness": 0.60,
            "speechiness": 0.04,
            "tempo": 105.0,
            "loudness": -14.0,
            "liveness": 0.18,
        },
    ),
    "workout": AgentMode(
        name="Workout",
        description="Fast and energetic music for exercise.",
        target={
            "danceability": 0.58,
            "energy": 0.90,
            "valence": 0.55,
            "acousticness": 0.08,
            "instrumentalness": 0.08,
            "speechiness": 0.05,
            "tempo": 145.0,
            "loudness": -4.5,
            "liveness": 0.18,
        },
    ),
    "feelgood": AgentMode(
        name="Feel-Good",
        description="Positive and danceable music for everyday listening.",
        target={
            "danceability": 0.76,
            "energy": 0.70,
            "valence": 0.82,
            "acousticness": 0.20,
            "instrumentalness": 0.04,
            "speechiness": 0.05,
            "tempo": 112.0,
            "loudness": -6.0,
            "liveness": 0.18,
        },
    ),
    "rhythm": AgentMode(
        name="Rhythm / Urban",
        description="Highly danceable tracks with stronger spoken-word content.",
        target={
            "danceability": 0.86,
            "energy": 0.70,
            "valence": 0.55,
            "acousticness": 0.10,
            "instrumentalness": 0.03,
            "speechiness": 0.23,
            "tempo": 96.0,
            "loudness": -6.0,
            "liveness": 0.18,
        },
    ),
}


class SmartPlaylistAgent:
    """A non-chatbot agent that selects music using a learned K-Means model."""

    def __init__(self, csv_path: Path, output_dir: Path) -> None:
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data = pd.read_csv(csv_path)
        self.clean_data = self.data.dropna(subset=FEATURES).copy()

        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.clean_data[FEATURES])

        self.model = KMeans(
            n_clusters=4,
            n_init=20,
            random_state=42,
        )
        self.labels = self.model.fit_predict(self.scaled_features)
        self.clean_data["cluster"] = self.labels

        self.cluster_centers_raw = pd.DataFrame(
            self.scaler.inverse_transform(self.model.cluster_centers_),
            columns=FEATURES,
        )
        self.mode_to_cluster = self._map_modes_to_clusters()

    def _map_modes_to_clusters(self) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        for mode_key, mode in MODES.items():
            target_frame = pd.DataFrame([mode.target], columns=FEATURES)
            target_scaled = self.scaler.transform(target_frame)[0]
            distances = np.linalg.norm(self.model.cluster_centers_ - target_scaled, axis=1)
            mapping[mode_key] = int(np.argmin(distances))
        return mapping

    def recommend(
        self,
        mode_key: str,
        count: int = 10,
        allow_explicit: bool = True,
    ) -> pd.DataFrame:
        if mode_key not in MODES:
            valid = ", ".join(MODES)
            raise ValueError(f"Unknown mode '{mode_key}'. Choose one of: {valid}")

        mode = MODES[mode_key]
        cluster_id = self.mode_to_cluster[mode_key]

        target_frame = pd.DataFrame([mode.target], columns=FEATURES)
        target_scaled = self.scaler.transform(target_frame)[0]

        candidates = self.clean_data[self.clean_data["cluster"] == cluster_id].copy()
        if not allow_explicit and "explicit" in candidates.columns:
            candidates = candidates[~candidates["explicit"]].copy()

        candidate_scaled = self.scaler.transform(candidates[FEATURES])
        candidates["distance_to_mode"] = np.linalg.norm(
            candidate_scaled - target_scaled,
            axis=1,
        )

        display_columns = [
            "track_name",
            "artist_name",
            "genre",
            "popularity",
            "danceability",
            "energy",
            "valence",
            "acousticness",
            "tempo",
            "cluster",
            "distance_to_mode",
        ]
        return (
            candidates.sort_values("distance_to_mode")
            .head(count)[display_columns]
            .reset_index(drop=True)
        )

    def create_outputs(self) -> Dict[str, float]:
        self._create_correlation_matrix()
        self._create_elbow_chart()
        pca_variance = self._create_pca_chart()
        self._create_cluster_profile_chart()
        self._create_cluster_size_chart()

        sample_size = min(2500, len(self.clean_data))
        sample_indices = np.random.default_rng(42).choice(
            len(self.clean_data),
            size=sample_size,
            replace=False,
        )
        silhouette = silhouette_score(
            self.scaled_features[sample_indices],
            self.labels[sample_indices],
        )

        cluster_sizes = (
            self.clean_data["cluster"].value_counts().sort_index().rename("tracks")
        )
        summary = self.cluster_centers_raw.copy()
        summary.insert(0, "tracks", cluster_sizes)
        summary.index.name = "cluster"
        summary.to_csv(self.output_dir / "cluster_summary.csv")

        recommendation_frames: List[pd.DataFrame] = []
        for mode_key, mode in MODES.items():
            recommendations = self.recommend(mode_key, count=5)
            recommendations.insert(0, "mode", mode.name)
            recommendation_frames.append(recommendations)
        pd.concat(recommendation_frames, ignore_index=True).to_csv(
            self.output_dir / "sample_recommendations.csv",
            index=False,
        )

        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "features": FEATURES,
                "mode_to_cluster": self.mode_to_cluster,
            },
            self.output_dir / "smart_playlist_model.joblib",
        )

        results = {
            "rows": float(len(self.clean_data)),
            "silhouette": float(silhouette),
            "pca_variance": float(pca_variance),
        }
        self._write_results_summary(results, cluster_sizes)
        return results

    def _create_correlation_matrix(self) -> None:
        correlation = self.clean_data[FEATURES].corr()
        figure, axis = plt.subplots(figsize=(10, 8))
        image = axis.imshow(correlation, vmin=-1, vmax=1)
        axis.set_xticks(range(len(FEATURES)))
        axis.set_yticks(range(len(FEATURES)))
        axis.set_xticklabels(FEATURES, rotation=45, ha="right")
        axis.set_yticklabels(FEATURES)
        for row in range(len(FEATURES)):
            for column in range(len(FEATURES)):
                axis.text(
                    column,
                    row,
                    f"{correlation.iloc[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        figure.colorbar(image, ax=axis, label="Correlation")
        axis.set_title("Correlation Matrix of Spotify Audio Features")
        figure.tight_layout()
        figure.savefig(self.output_dir / "correlation_matrix.png", dpi=180)
        plt.close(figure)

    def _create_elbow_chart(self) -> None:
        sample_size = min(10000, len(self.clean_data))
        sample_indices = np.random.default_rng(42).choice(
            len(self.clean_data),
            size=sample_size,
            replace=False,
        )
        sample = self.scaled_features[sample_indices]

        k_values = list(range(1, 11))
        inertias: List[float] = []
        for k in k_values:
            model = KMeans(n_clusters=k, n_init=10, random_state=42)
            model.fit(sample)
            inertias.append(float(model.inertia_))

        figure, axis = plt.subplots(figsize=(9, 5.5))
        axis.plot(k_values, inertias, marker="o")
        axis.set_title("Elbow Method: Four Clusters Are a Practical Choice")
        axis.set_xlabel("Number of Clusters (K)")
        axis.set_ylabel("Inertia")
        axis.set_xticks(k_values)
        axis.grid(True, alpha=0.35)
        figure.tight_layout()
        figure.savefig(self.output_dir / "elbow_method.png", dpi=180)
        plt.close(figure)

    def _create_pca_chart(self) -> float:
        pca = PCA(n_components=2)
        coordinates = pca.fit_transform(self.scaled_features)
        explained = float(pca.explained_variance_ratio_.sum())

        figure, axis = plt.subplots(figsize=(10, 6.5))
        scatter = axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=self.labels,
            s=7,
            alpha=0.30,
        )
        axis.set_title("K-Means Music Clusters in 2D PCA Space")
        axis.set_xlabel("Principal Component 1")
        axis.set_ylabel("Principal Component 2")
        axis.legend(*scatter.legend_elements(), title="Cluster", loc="upper right")
        axis.text(
            0.01,
            0.02,
            f"First two components explain {explained:.1%} of variance",
            transform=axis.transAxes,
            fontsize=10,
        )
        figure.tight_layout()
        figure.savefig(self.output_dir / "pca_clusters.png", dpi=180)
        plt.close(figure)
        return explained

    def _create_cluster_profile_chart(self) -> None:
        standardized_centers = pd.DataFrame(
            self.model.cluster_centers_,
            columns=FEATURES,
        )
        figure, axis = plt.subplots(figsize=(11, 4.8))
        image = axis.imshow(standardized_centers, aspect="auto", vmin=-2.2, vmax=2.2)
        axis.set_xticks(range(len(FEATURES)))
        axis.set_xticklabels(FEATURES, rotation=35, ha="right")
        axis.set_yticks(range(4))
        axis.set_yticklabels([f"Cluster {index}" for index in range(4)])
        for row in range(4):
            for column in range(len(FEATURES)):
                axis.text(
                    column,
                    row,
                    f"{standardized_centers.iloc[row, column]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        figure.colorbar(image, ax=axis, label="Standardized value")
        axis.set_title("Cluster Profiles: Values Above or Below the Dataset Average")
        figure.tight_layout()
        figure.savefig(self.output_dir / "cluster_profiles.png", dpi=180)
        plt.close(figure)

    def _create_cluster_size_chart(self) -> None:
        counts = self.clean_data["cluster"].value_counts().sort_index()
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar(counts.index.astype(str), counts.values)
        axis.set_title("Number of Tracks in Each Cluster")
        axis.set_xlabel("Cluster")
        axis.set_ylabel("Tracks")
        for index, value in enumerate(counts.values):
            axis.text(index, value, f"{value:,}", ha="center", va="bottom")
        figure.tight_layout()
        figure.savefig(self.output_dir / "cluster_sizes.png", dpi=180)
        plt.close(figure)

    def _write_results_summary(
        self,
        results: Dict[str, float],
        cluster_sizes: pd.Series,
    ) -> None:
        lines = [
            "SMART PLAYLIST AGENT - MODEL RESULTS",
            "====================================",
            f"Tracks used: {int(results['rows']):,}",
            "Features: " + ", ".join(FEATURES),
            "Model: K-Means (K=4, n_init=20, random_state=42)",
            "Preprocessing: StandardScaler",
            f"Silhouette score (2,500-track sample): {results['silhouette']:.3f}",
            f"PCA variance explained by first two components: {results['pca_variance']:.1%}",
            "",
            "Cluster sizes:",
        ]
        for cluster_id, value in cluster_sizes.items():
            lines.append(f"  Cluster {cluster_id}: {int(value):,} tracks")
        lines.extend(["", "Mode-to-cluster decisions:"])
        for mode_key, cluster_id in self.mode_to_cluster.items():
            lines.append(f"  {MODES[mode_key].name}: Cluster {cluster_id}")
        (self.output_dir / "results_summary.txt").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and demonstrate the Smart Playlist Agent.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("spotify_tracks.csv"),
        help="Path to spotify_tracks.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Folder for charts, summaries, and the saved model",
    )
    parser.add_argument(
        "--mode",
        choices=list(MODES),
        default="workout",
        help="Mode to demonstrate after training",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of songs to recommend",
    )
    parser.add_argument(
        "--no-explicit",
        action="store_true",
        help="Exclude explicit tracks from recommendations",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    agent = SmartPlaylistAgent(arguments.csv, arguments.output)
    results = agent.create_outputs()

    selected_mode = MODES[arguments.mode]
    recommendations = agent.recommend(
        arguments.mode,
        count=arguments.count,
        allow_explicit=not arguments.no_explicit,
    )

    print("\nSmart Playlist Agent")
    print("====================")
    print(f"Mode: {selected_mode.name}")
    print(f"Decision: use Cluster {agent.mode_to_cluster[arguments.mode]}")
    print(selected_mode.description)
    print(f"Silhouette score: {results['silhouette']:.3f}")
    print("\nRecommended tracks:")
    print(recommendations[["track_name", "artist_name", "genre", "energy", "tempo"]].to_string(index=False))
    print(f"\nOutputs saved to: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
