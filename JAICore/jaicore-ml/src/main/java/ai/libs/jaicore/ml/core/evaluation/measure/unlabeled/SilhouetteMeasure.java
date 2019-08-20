package ai.libs.jaicore.ml.core.evaluation.measure.unlabeled;


import ai.libs.jaicore.ml.tsc.util.MathUtil;
import java.util.List;

public class SilhouetteMeasure extends AInternalClusteringValidationMeasure {

	@Override
	public Double calculateMeasure(final List<List<double[]>> clusters, final List<double[]> centroids, final boolean erasure) {
		double sum = 0;
		double size = 0;
		for (int i = 0; i < clusters.size(); i++) {
			final List<double[]> cluster = clusters.get(i);
			size += cluster.size();
			for (int j = 0; j < cluster.size(); j++) {
				final double[] point = cluster.get(j);
				final double distanceToOwnCluster = distancePointToCluster(cluster, point);
				double distanceToOtherCluster = Double.MAX_VALUE;
				for (int k = 0; k < clusters.size(); k++) {
					if (k == i) {
						continue;
					}
					final double tmp = distancePointToCluster(clusters.get(k), point);
					if (tmp < distanceToOtherCluster) {
						distanceToOtherCluster = tmp;
					}
				}
				if (distanceToOtherCluster != 0 && distanceToOwnCluster != 0) {
					sum += (distanceToOtherCluster - distanceToOwnCluster) / Math.max(distanceToOtherCluster, distanceToOwnCluster);
				}
			}
		}
		return -sum / size;
	}

	private double distancePointToCluster(final List<double[]> cluster, final double[] point) {
		double sum = 0;
		for (int i = 0; i < cluster.size(); i++) {
			sum += Math.sqrt(MathUtil.singleSquaredEuclideanDistance(cluster.get(i), point));
		}
		return sum / cluster.size();
	}

}
