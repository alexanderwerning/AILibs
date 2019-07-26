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

	private final List<Double> scatter = new ArrayList<>();
	private List<List<double[]>> clusters;

	@Override
	public Double calculateMeasure(final List<List<double[]>> clusters) {
		this.clusters = clusters;
		final int c = clusters.size();

		for (int i = 0; i < c; i++) {
			this.scatter.add(sumOfDistances(clusters.get(i), this.centroids.get(i)));
		}

		double max_D = 0;
		for (int i = 0; i < c; i++) {
			final double cur_D = D3(i);
			if (cur_D > max_D) {
				max_D = cur_D;
			}
		}

		double min_d = 0;
		for (int i = 0; i < c; i++) {
			for (int j = 0; j < c; j++) {
				if (i == j) {
					continue;
				}
				final double cur_d = d5(i, j);
				if (cur_d < min_d) {
					min_d = cur_d;
				}
			}
		}
		final double originalMeasure = min_d / max_D;
		return -(originalMeasure);
	}

	private double D3(final int i) {
		return 2 * this.scatter.get(i) / this.clusters.get(i).size();
	}

	private double d5(final int i, final int j) {
		final double size = this.clusters.get(i).size() + this.clusters.get(j).size();
		double sum = 0;
		sum += sumOfDistances(this.clusters.get(i), this.centroids.get(i));
		sum += sumOfDistances(this.clusters.get(j), this.centroids.get(j));
		return sum / size;
	}

	private double sumOfDistances(final List<double[]> cluster, final double[] point) {
		double sum = 0;
		for (int i = 0; i < cluster.size(); i++) {
			sum += singleSquaredEuclideanDistance(cluster.get(i), point);
		}
		return sum;
	}
}
