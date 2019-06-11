package jaicore.ml.core.evaluation.measure.unlabeled;

import java.util.List;

import jaicore.basic.aggregate.IAggregateFunction;

public class DensityBasedMeasure extends AInternalClusteringMeasure{


	@Override
	public Double calculateMeasure(final List<List<double[]>> clusters) {
		return 0d;
	}
}
