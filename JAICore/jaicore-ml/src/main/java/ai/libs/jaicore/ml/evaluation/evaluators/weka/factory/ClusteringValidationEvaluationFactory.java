package ai.libs.jaicore.ml.evaluation.evaluators.weka.factory;

import ai.libs.jaicore.ml.evaluation.evaluators.weka.ClusteringValidationEvaluator;
import ai.libs.jaicore.ml.evaluation.evaluators.weka.IClassifierEvaluator;
import weka.core.Instances;

public class ClusteringValidationEvaluationFactory implements IClassifierEvaluatorFactory{

	@Override
	public IClassifierEvaluator getIClassifierEvaluator(final Instances dataset, final long seed) throws ClassifierEvaluatorConstructionFailedException {
		return new ClusteringValidationEvaluator(dataset);
	}
}
