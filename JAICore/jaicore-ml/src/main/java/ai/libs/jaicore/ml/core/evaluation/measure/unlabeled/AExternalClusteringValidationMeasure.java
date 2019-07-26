package ai.libs.jaicore.ml.core.evaluation.measure.unlabeled;

import ai.libs.jaicore.basic.aggregate.IAggregateFunction;
import ai.libs.jaicore.ml.ClusterUtil;
import java.util.List;

public abstract class AExternalClusteringValidationMeasure implements IExternalClusteringValidationMeasure {

	public abstract Double calculateExternalMeasure(List<Double> actual, List<Double> expected, int[][] contingency);

	public Double calculateExternalMeasure(final List<Double> actual, final List<Double> expected) {
		return calculateExternalMeasure(actual, expected, ClusterUtil.calculateContingencyMatrix(actual, expected));
	}

	@Override
	public Double calculateMeasure(final Double actual, final Double expected) {
		return 0d;
	}

	@Override
	public Double calculateMeasure(final List<Double> actual, final List<Double> expected, final IAggregateFunction<Double> aggregateFunction) {
		return null;
	}

	@Override
	public Double calculateAvgMeasure(final List<Double> actual, final List<Double> expected) {
		return null;
	}

	@Override
	public List<Double> calculateMeasure(final List<Double> actual, final List<Double> expected) {
		return null;
	}
}
