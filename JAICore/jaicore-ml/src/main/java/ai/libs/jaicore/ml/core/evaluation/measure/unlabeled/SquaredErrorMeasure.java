package ai.libs.jaicore.ml.core.evaluation.measure.unlabeled;

import ai.libs.jaicore.ml.tsc.distances.EuclideanDistance;
import java.util.List;

public class SquaredErrorMeasure extends AInternalClusteringValidationMeasure {

	@Override
	public Double calculateMeasure(final List<List<double[]>> clusters, final List<double[]> centroids, final boolean erasure) {
		double sum = 0;

		for (int i = 0; i < clusters.size(); i++) {

			final double[] centroid = centroids.get(i);
			for (final double[] instance : clusters.get(i)) {
				sum += new EuclideanDistance().distance(centroid, instance);
			}
		}
		return sum;
	}
}
