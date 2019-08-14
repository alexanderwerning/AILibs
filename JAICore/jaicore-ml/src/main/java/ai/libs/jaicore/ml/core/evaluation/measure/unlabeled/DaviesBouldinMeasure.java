package ai.libs.jaicore.ml.core.evaluation.measure.unlabeled;


import static ai.libs.jaicore.ml.tsc.util.MathUtil.singleSquaredEuclideanDistance;

import ai.libs.jaicore.ml.ClusterUtil;
import java.util.LinkedList;
import java.util.List;

public class DaviesBouldinMeasure extends AInternalClusteringValidationMeasure {

	/**
	 * In the original paper this is formulated as a loss (minimized for best clustering) loss measure is needed here
	 */
	@Override
	public Double calculateMeasure(final List<List<double[]>> clusters) {
		System.out.println("number of clusters: " + clusters.size());
		final int N = clusters.size();
		final List<double[]> centroids = new LinkedList<>();
		for (int i = 0; i < N; i++) {
			centroids.add(ClusterUtil.calculateCentroid(clusters.get(i)));
		}
		double sum = 0;
		for (int i = 0; i < N; i++) {
			double max = 0;
			for (int j = 0; j < N; j++) {
				if (i != j) {
					final double rij = (calculateDispersion(clusters.get(i), centroids.get(i)) + calculateDispersion(clusters.get(j), centroids.get(j))) / clusterDistances(clusters.get(i),
						centroids.get(i), clusters.get(j), centroids.get(j));
					if (rij > max) {
						max = rij;
					}
				}
			}
			sum += max;
		}
		return sum / N;
	}

	private double clusterDistances(final List<double[]> cluster1, final double[] centroid1, final List<double[]> cluster2, final double[] centroid2) {
		return Math.sqrt(singleSquaredEuclideanDistance(centroid1, centroid2));
	}


	private double calculateDispersion(final List<double[]> cluster, final double[] centroid) {
		// calculate average distance from centroid
		double sqsum = 0;
		for (int i = 0; i < cluster.size(); i++) {
			double vectorLength = 0;
			for (int j = 0; j < centroid.length; j++) {
				vectorLength += Math.pow(centroid[j] - cluster.get(i)[j], 2);
			}
			sqsum += vectorLength;
		}
		sqsum /= cluster.size();
		return Math.sqrt(sqsum);
	}
}
