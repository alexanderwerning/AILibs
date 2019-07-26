package ai.libs.jaicore.ml.evaluation.evaluators.weka.factory;

import ai.libs.jaicore.ml.core.evaluation.measure.IMeasure;
import ai.libs.jaicore.ml.evaluation.evaluators.weka.ClusteringValidationEvaluator;
import ai.libs.jaicore.ml.evaluation.evaluators.weka.IClassifierEvaluator;
import weka.core.Instances;

public class ClusteringValidationEvaluationFactory implements IClassifierEvaluatorFactory{
	IMeasure<Instances, Double> measure;
	@Override
	public IClassifierEvaluator getIClassifierEvaluator(final Instances dataset, final long seed) throws ClassifierEvaluatorConstructionFailedException {
		if(measure == null) {
			throw new IllegalStateException("Cannot create Clustering Validation Evaluator, because no measure has been set!");
		}
		return new ClusteringValidationEvaluator(measure,dataset);
	}

	public ClusteringValidationEvaluationFactory withMeasure(IMeasure<Instances, Double> measure){
		this.measure = measure;
		return this;
	}
}
