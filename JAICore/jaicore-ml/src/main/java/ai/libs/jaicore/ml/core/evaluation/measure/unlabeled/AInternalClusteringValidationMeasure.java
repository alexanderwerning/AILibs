package ai.libs.jaicore.ml.core.evaluation.measure.unlabeled;


import ai.libs.jaicore.basic.aggregate.IAggregateFunction;
import ai.libs.jaicore.ml.ClusterUtil;
import java.util.ArrayList;
import java.util.List;
import weka.core.Instances;

public abstract class AInternalClusteringValidationMeasure implements IInternalClusteringValidationMeasure {


	public abstract Double calculateMeasure(List<List<double[]>> clusters, List<double[]> centroids, boolean erasure);// erasure forbids two list parameters

	private Double calculateMeasureAndCentroids(final List<List<double[]>> clusters) {
		return calculateMeasure(clusters, ClusterUtil.calculateCentroids(clusters), false);
	}

	@Override
	public Double calculateMeasure(final Instances labeled, final Instances nul) {
		final List<List<double[]>> clusters = ClusterUtil.separateClusters(labeled);
		final double measureResult = calculateMeasureAndCentroids(clusters);
		
		if (clusters.size() == labeled.size() || clusters.size() == 0 || clusters.size() == 1) {
			return Double.MAX_VALUE;
		}
		return measureResult;
	}

	@Override
	public List<Double> calculateMeasure(final List<Instances> actual, final List<Instances> expected) {
		if (actual.size() != expected.size()) {
			throw new IllegalArgumentException("actual and expected lists must have the same length");
		}
		final List<Double> results = new ArrayList<>();
		for (int i = 0; i < actual.size(); i++) {
			results.add(this.calculateMeasure(actual.get(i), expected.get(i)));
		}
		return results;
	}

	@Override
	public Double calculateMeasure(final List<Instances> actual, final List<Instances> expected, final IAggregateFunction<Double> aggregateFunction) {
		return null;
	}

	@Override
	public Double calculateAvgMeasure(final List<Instances> actual, final List<Instances> expected) {
		return null;
	}
}