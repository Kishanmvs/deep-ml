def k_means_clustering(
    points: list[tuple[float, ...]],
    k: int,
    initial_centroids: list[tuple[float, ...]],
    max_iterations: int
) -> list[tuple[float, ...]]:

    centroids = [list(c) for c in initial_centroids]

    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]

        # Assign each point to the nearest centroid
        for point in points:
            distances = [
                sum((point[d] - centroids[i][d]) ** 2
                    for d in range(len(point)))
                for i in range(k)
            ]

            cluster_idx = distances.index(min(distances))
            clusters[cluster_idx].append(point)

        # Update centroids
        new_centroids = []

        for i in range(k):
            if clusters[i]:
                centroid = [
                    sum(point[d] for point in clusters[i]) / len(clusters[i])
                    for d in range(len(centroids[i]))
                ]
                new_centroids.append(centroid)
            else:
                new_centroids.append(centroids[i])

        centroids = new_centroids

    final_centroids = [
        tuple(round(value, 4) for value in centroid)
        for centroid in centroids
    ]

    return final_centroids