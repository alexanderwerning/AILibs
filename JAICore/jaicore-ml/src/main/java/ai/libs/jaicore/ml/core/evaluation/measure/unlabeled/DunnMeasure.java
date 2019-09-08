package ai.libs.jaicore.ml.core.evaluation.measure.unlabeled;

import static ai.libs.jaicore.ml.tsc.util.MathUtil.singleSquaredEuclideanDistance;

import java.util.ArrayList;
import java.util.List;

public class DunnMeasure extends AInternalClusteringValidationMeasure {

	/**
	 * improved version, taken from Some New Indexes of Cluster Validity James C. Bezdek, Fellow, IEEE, and Nikhil R. Pal
	 *
	 * within cluster scatter measure D3, average distance to sample mean cluster separation measure d5, average point to sample mean distance
	 *
	 * is maximized in the original version, so we negate it here
	 */


	@Override
	public Double calculateMeasure(final List<List<double[]>> clusters, final List<double[]> centroids, final boolean erasure) {
		final int c = clusters.size();
		final List<Double> scatter = new ArrayList<>();
		for (int i = 0; i < c; i++) {
			scatter.add(sumOfDistances(clusters.get(i), centroids.get(i)));
		}

		double max_D = 0;
		for (int i = 0; i < c; i++) {
			final double cur_D = D3(clusters.get(i), scatter.get(i));
			if (cur_D > max_D) {
				max_D = cur_D;
			}
		}

		double min_d = Double.MAX_VALUE;
		for (int i = 0; i < c; i++) {
			for (int j = 0; j < c; j++) {
				final double cur_d = d5(clusters, centroids, i, j);
				if (cur_d > 0 && cur_d < min_d) {
					min_d = cur_d;
				}
			}
		}
		double originalMeasure = min_d / max_D;
		if (max_D == 0) {
			originalMeasure = Double.MAX_VALUE;
		}
		return -(originalMeasure);
	}

	private double D3(final List<double[]> cluster, final double scatter) {
		return 2 * scatter / cluster.size();
	}

	private double d5(final List<List<double[]>> clusters, final List<double[]> centroids, final int i, final int j) {
		final double size = clusters.get(i).size() + clusters.get(j).size();
		double sum = 0;
		sum += sumOfDistances(clusters.get(i), centroids.get(i));
		sum += sumOfDistances(clusters.get(j), centroids.get(j));
		return sum / size;
	}

	private double sumOfDistances(final List<double[]> cluster, final double[] point) {
		double sum = 0;
		for (int i = 0; i < cluster.size(); i++) {
			sum += Math.sqrt(singleSquaredEuclideanDistance(cluster.get(i), point));
		}
		return sum;
	}
}
